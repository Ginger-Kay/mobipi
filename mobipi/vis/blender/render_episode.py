"""
This code takes a replayed episode and renders it out along with illustrations
of the optimization process using Blender. Rendering an episode takes ~1 hour.

Usage (run in Blender background):
    First, fill in the LOG_ROOT_DIR constant below the imports.
    Then, run the following script.

    blender --background --python render_episode.py -- \
        <usd_subdir> <output_dir> \
        [--debug] [--usd_start_frame 165]

    • usd_subdir: full path to one “replay_…_epX” directory
                 (e.g. “…/replay_CloseDrawer_layout0_style0_seed1_ep0”)
    • output_dir: directory where rendered PNGs will be written
    • --debug:    (optional) save a .blend file for debugging
    • --usd_start_frame: Blender frame at which to begin episode playback

This script will:
  1. Find the single USD file (highest frame index) in usd_subdir/usd/frames.
  2. Import that USD at its default time = 1 into Blender.
  3. Apply a built‐in frame offset on each Transform Cache constraint so that
     USD playback is delayed until Blender frame = usd_start_frame.
  4. Set up a camera that tracks the room’s center until frame T₀ = 30, then
     jumps to top‐down.
  5. Extract (env_name, layout_id, style_id, seed, ep_idx) from the subdir name.
  6. Animate BoT points using **sampled_points** & **sampled_scores** from the
     original episode's ep?_info.npz under LOG_ROOT_DIR, with explicit keyframes
     controlling visibility and alpha.
  7. Draw a tube‐shaped navigation path over 30 frames using
     **base_pos_history** from the replay episode's ep?_info.npz inside
     usd_subdir (with keyframes on bevel_factor_end), then zoom camera back over
     30 frames.
  8. Starting at Blender frame = usd_start_frame, animate the navigation‐path's
     bevel_factor_start from 0→1 (with keyframes), erasing the path behind the
     robot (the robot’s actual transform is now handled by the USD constraint +
     frame offset).
  9. Render all frames into `<output_dir>/frame_####.png`.

@yjy0625, with lots of help from ChatGPT
"""

import bpy
import sys
import subprocess
import os
import math
import argparse
import re
import numpy as np
from glob import glob
from pxr import Usd, UsdGeom, Sdf
from mathutils import Vector, Matrix


LOG_ROOT_DIR = "[FILL IN THIS]"
if LOG_ROOT_DIR == "[FILL IN THIS]":
    print(f"Please fill in [LOG_ROOT_DIR] before proceeding!")
    exit(0)


# Attempt to import tqdm; if missing, install it
try:
    from tqdm import tqdm
except ImportError:
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# ------------------------------------------------------------------------------------------------
# 1. PROGRESS BAR UTIL
# ------------------------------------------------------------------------------------------------
def print_progress(current, total, prefix='', length=40):
    percent = current / total if total else 1
    filled = int(length * percent)
    bar = '#' * filled + '-' * (length - filled)
    sys.stdout.write(f"\r{prefix}: |{bar}| {current}/{total}")
    sys.stdout.flush()
    if current == total:
        print()

# ------------------------------------------------------------------------------------------------
# 2. ARGUMENT PARSING
# ------------------------------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    if '--' not in argv:
        sys.exit("Error: script arguments must follow '--'")
    idx = argv.index('--') + 1
    parser = argparse.ArgumentParser(
        description="Single‐room USD + BoT optimization + navigation‐path animation"
    )
    parser.add_argument('usd_subdir', help='Path to one “replay_…_epX” directory')
    parser.add_argument('output_dir', help='Directory to save renders')
    parser.add_argument('--usd_start_frame', type=int, default=165,
                        help='Blender frame at which to begin USD replay (default=165)')
    parser.add_argument('--debug', action='store_true', help='Save .blend for debug')
    return parser.parse_args(argv[idx:])

# ------------------------------------------------------------------------------------------------
# 3. EXTRACT METADATA FROM SUBDIR NAME
# ------------------------------------------------------------------------------------------------
def parse_subdir_name(basename):
    """
    Expect format: "replay_<Env>_layout<Li>_style<Si>_seed<Se>_ep<Ep>"
    Example: "replay_CloseDrawer_layout0_style0_seed1_ep0"
    Returns: (env_name:str, layout_id:int, style_id:int, seed:int, ep_idx:int)
    """
    patt = r"replay_([^_]+)_layout(\d+)_style(\d+)_seed(\d+)_ep(\d+)"
    m = re.match(patt, basename)
    if not m:
        raise RuntimeError(f"Could not parse subdir name '{basename}' with pattern {patt}")
    env_name   = m.group(1)
    layout_id  = int(m.group(2))
    style_id   = int(m.group(3))
    seed       = int(m.group(4))
    ep_idx     = int(m.group(5))
    return env_name, layout_id, style_id, seed, ep_idx

# ------------------------------------------------------------------------------------------------
# 4. IMPORT A SINGLE USD FILE (time=1)
# ------------------------------------------------------------------------------------------------
def import_single_usd(usd_filepath):
    """
    Import one USD file into the current scene at its default time sample (time=1).
    Return: set of newly‐added objects.
    """
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=usd_filepath)
    new_objs = set(bpy.data.objects) - before
    for o in list(new_objs):
        if o.type in {'LIGHT', 'CAMERA'}:
            bpy.data.objects.remove(o, do_unlink=True)
    new_objs = set(bpy.data.objects) - before
    return new_objs

# ------------------------------------------------------------------------------------------------
# 5. APPLY TRANSFORM CACHE FRAME OFFSET
# ------------------------------------------------------------------------------------------------
def apply_transform_cache_offset(imported_objects, usd_start_frame):
    """
    Look through all imported_objects. Whenever you find a constraint
    whose type is 'TRANSFORM_CACHE' (Blender 4.x USD importer),
    set its built‐in frame_offset to −(usd_start_frame − 1). That way,
    USD playback stays frozen until Blender frame = usd_start_frame.
    """
    offset_value = usd_start_frame - 1

    for obj in imported_objects:
        for c in obj.constraints:
            if c.type == 'TRANSFORM_CACHE':
                c.cache_file.frame_offset = offset_value


def apply_visibility_from_usd(imported_objects, usd_filepath, usd_start_frame):
    """
    1) Open the USD file with pxr.Usd.Stage.Open().
    2) For each `obj` that has a TRANSFORM_CACHE constraint, pull out
       c.cache_file.prim_path (e.g. "/World/Mesh_Xform_sink_main_group_water_id29").
    3) Query that prim’s 'visibility' attribute’s time samples (if any).
    4) For each (time_sample, token) where token is not the final “invisible”,
       insert a keyframe on obj.hide_viewport & obj.hide_render at:
           blender_frame = int(time_sample) + (usd_start_frame – 1)
       using hide=True if token == 'invisible', or False if token == 'inherited'.
    5) Skip any “invisible” sample at the very end so objects don’t vanish on the last frame.
    """

    # Open the USD stage once
    stage = Usd.Stage.Open(usd_filepath)
    if not stage:
        print(f"❌  Could not open USD: {usd_filepath!r}")
        return

    # Pre‐compute offset so we do exactly the same mapping as transform_cache does:
    offset_value = usd_start_frame - 1

    for obj in imported_objects:
        # Find the TRANSFORM_CACHE constraint so we can read prim_path
        prim_path_str = None
        for c in obj.constraints:
            if c.type == 'TRANSFORM_CACHE':
                prim_path_str = c.object_path
                break
        if not prim_path_str:
            # No TRANSFORM_CACHE, skip
            continue

        prim = stage.GetPrimAtPath(prim_path_str)
        if not prim or not prim.IsValid():
            # couldn’t find that prim in USD
            continue

        # Fetch the “visibility” attribute on that prim:
        vis_attr = prim.GetAttribute('visibility')
        if not vis_attr:
            # No visibility attr was authored → assume always visible
            # (you could insert a single keyframe: hide=False at frame=usd_start_frame)
            continue

        # Get all time samples for visibility:
        try:
            time_samples = vis_attr.GetTimeSamples()
        except Exception:
            # Fallback if GetTimeSamples fails:
            bracket = vis_attr.GetBracketingTimeSamples(0.0)
            time_samples = list(bracket) if bracket else []
        time_samples = sorted(time_samples)

        if not time_samples:
            # No time samples → maybe it was static “inherited” or “invisible.”
            static_val = vis_attr.Get()
            # If static_val == "invisible", we could hide it from the very start.
            # But most USDs have at least [0.0: 'inherited' ] if always visible.
            if static_val == 'invisible':
                # Hide every frame (or at least from frame=1):
                obj.hide_viewport = True
                obj.hide_render   = True
                obj.keyframe_insert(data_path="hide_viewport", frame=1)
                obj.keyframe_insert(data_path="hide_render",   frame=1)
            continue

        # If there *are* multiple samples, we skip the final sample if it is 'invisible'
        max_time = max(time_samples)

        for t in time_samples:
            val = vis_attr.Get(t)
            if (t == max_time and val == 'invisible'):
                # Skip the final “invisible” so things don’t vanish on the last frame
                continue

            # Compute Blender frame exactly as transform_cache does:
            blender_frame = int(t + offset_value)

            # Map USD token → hide boolean:
            hide_flag = (val == 'invisible')

            # Insert keyframes for hide_viewport & hide_render
            obj.hide_viewport = hide_flag
            obj.hide_render   = hide_flag
            obj.keyframe_insert(data_path="hide_viewport", frame=blender_frame)
            obj.keyframe_insert(data_path="hide_render",   frame=blender_frame)

    print(f"✅  Applied USD visibility keyframes to {len(imported_objects)} object(s).")

# ------------------------------------------------------------------------------------------------
# 6. FILTER + RETOUCH MESHES
# ------------------------------------------------------------------------------------------------
def filter_and_retouch(mesh_objects, remove_subs):
    """
    - Remove any MESH whose name contains any of remove_subs.
    - On kept meshes:
      • If a material’s Base Color ≈(0.251,0.251,0.251), change it to (0.08,0.08,0.08).
      • If Base Color ≈(1,1,1), set Metallic=1.0, Roughness=0.35 on that Principled BSDF.
      • Smooth normals + add EdgeSplit(angle=30°).
    Returns list of kept objects.
    """
    kept = []
    for obj in mesh_objects:
        if obj.type != 'MESH' or any(sub in obj.name for sub in remove_subs):
            continue
        to_remove = False
        for slot in obj.material_slots:
            mat = slot.material
            if mat and hasattr(mat, 'diffuse_color') and tuple(mat.diffuse_color) in [
                (0.5, 0, 0, 1),
                (0, 0.5, 0, 1),
                (0, 1, 0, 1)
            ]:
                to_remove = True
                break
        if to_remove:
            continue
        kept.append(obj)

    for obj in kept:
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not bsdf:
                continue
            base = bsdf.inputs['Base Color'].default_value
            # If ≈(0.251,0.251,0.251):
            if all(abs(base[i] - 0.251) < 0.01 for i in range(3)):
                bsdf.inputs['Base Color'].default_value = (0.08, 0.08, 0.08, 1.0)
            # If white:
            if all(abs(base[i] - 1.0) < 1e-6 for i in range(3)):
                bsdf.inputs['Metallic'].default_value  = 1.0
                bsdf.inputs['Roughness'].default_value = 0.35

    for obj in kept:
        if obj.type != "MESH":
            continue
        for poly in obj.data.polygons:
            poly.use_smooth = True
        mod = obj.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
        mod.use_edge_angle = True
        mod.split_angle = math.pi / 6

    return kept

# ------------------------------------------------------------------------------------------------
# 7. COMPUTE FLOOR OFFSET
# ------------------------------------------------------------------------------------------------
def compute_floor_offset(kept, x_tile=0.0, y_tile=0.0):
    """
    Find the mesh whose name matches 'Mesh_floor_room_g0_.*_geom', compute its world‐space center,
    then return a translation Vector = target − (center×100, 0).  (Usd is in cm, so multiply by 100.)
    Returns: (offset:Vector, floor_obj:Object, orig_center_world:Vector)
    """
    floor_obj = next((o for o in kept if re.match(r"Mesh_floor_room_g0_.*_geom", o.name)), None)
    if not floor_obj:
        raise RuntimeError("Floor mesh not found for offset computation")
    bbox_corners = [floor_obj.matrix_world @ Vector(c) for c in floor_obj.bound_box]
    center = sum(bbox_corners, Vector()) / len(bbox_corners)
    orig = Vector((center.x * 100, center.y * 100, 0))
    target = Vector((x_tile, y_tile, 0))
    return (target - orig), floor_obj, orig

# ------------------------------------------------------------------------------------------------
# 8. CAMERA RESET & TRANSITION
# ------------------------------------------------------------------------------------------------
def reset_camera(track_location, dx, dy, T0=30):
    """
    1. Remove any existing CAMERA.
    2. Create a new camera that TRACKS the given Empty 'track_location'.
    3. Place it at (track_x, track_y - dist, height), where dist=max(dx,dy), angle=35° above horizon.
    4. Keyframe so that:
       • Frames 1..T0−1: “tracking” mode.
       • Frame =T0: jumps to top‐down.
    Returns: the new camera object.
    """
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    dist = max(dx, dy)
    angle = math.radians(35)
    height = dist * math.tan(angle)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = 35
    cam.data.clip_start = 0.5
    bpy.context.scene.camera = cam

    cam.location = (
        track_location.location.x,
        track_location.location.y - dist,
        height
    )
    track = cam.constraints.new(type='TRACK_TO')
    track.target = track_location
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    cam.keyframe_insert(data_path="location", frame=1)
    cam.keyframe_insert(data_path="rotation_euler", frame=1)

    cam.location = (
        track_location.location.x,
        track_location.location.y,
        dist * 1.18
    )
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    cam.keyframe_insert(data_path="location", frame=T0)
    cam.keyframe_insert(data_path="rotation_euler", frame=T0)

    return cam

# ------------------------------------------------------------------------------------------------
# 9. CREATE “CENTER” EMPTY
# ------------------------------------------------------------------------------------------------
def create_center_empty(kept):
    """
    Make an Empty at the floor’s world‐space center (scaled×100).
    Returns: that Empty.
    """
    empty = bpy.data.objects.new("RoomCenter", None)
    empty.empty_display_size = 0.5
    bpy.context.collection.objects.link(empty)
    empty.location = Vector((0.0, 0.0, 1.0))
    return empty

# ------------------------------------------------------------------------------------------------
# 10. ANIMATE BoT POINTS (OLD NPZ)
# ------------------------------------------------------------------------------------------------
def animate_optimization(old_npz_path, offset, start_frame=50,
                         side_length=0.1, corner_radius=0.02,
                         height=0.2, default_z=0.1,
                         ninit=2500, k_best=40, N_RAND=30, N_BO=30, T_FADE=10):
    """
    New, two‐pass version:
    (1) PASS 1: Load data and figure out, for each sample_idx, exactly 
              which frame it first appears in top‐k, and when it should fade out.
    (2) PASS 2: Create one cube per sample_idx, then insert keyframes for
              hide_viewport, hide_render, and alpha at the computed frames.

    Returns: total number of animation frames (N_RAND + N_BO).
    """

    # ----------------------------------------
    # 10.1 LOAD OLD NPZ & EXTRACT POINTS + SCORES
    # ----------------------------------------
    data = np.load(old_npz_path, allow_pickle=True)
    pts  = data["sampled_points"]   # shape (N,3)
    scrs = data["sampled_scores"]   # shape (N,)
    total_pts = pts.shape[0]

    # The “global best” is the index of the maximum score overall:
    global_best_idx = int(np.argmax(scrs))
    global_max = float(scrs.max()) if total_pts > 0 else 1.0

    # Compute how many “cumulative” points are used at each random/BO frame:
    rand_counts = [int(round((i+1)*ninit / N_RAND)) for i in range(N_RAND)]
    bo_total    = total_pts - ninit
    bo_counts   = [ninit + int(round((j+1)*bo_total / N_BO)) for j in range(N_BO)]

    # All together, we’ll have total_frames = N_RAND + N_BO
    total_frames = N_RAND + N_BO


    viridis = [
        (0.9176, 0.3451, 0.0471),
        (0.9176, 0.3451, 0.0471)
    ]
    def viridis_color(norm_val):
        """
        Given norm_val in [0,1], interpolate linearly between the 11 Viridis control points.
        Returns: (r,g,b,a=1.0)
        """
        x = max(0.0, min(1.0, norm_val))
        idx = x * (len(viridis) - 1)
        i0 = math.floor(idx)
        i1 = min(i0 + 1, len(viridis) - 1)
        f  = idx - i0
        c0 = viridis[i0]
        c1 = viridis[i1]
        return (
            (1 - f) * c0[0] + f * c1[0],
            (1 - f) * c0[1] + f * c1[1],
            (1 - f) * c0[2] + f * c1[2],
            1.0
        )

    # ----------------------------------------
    # 10.2 PASS 1: FIGURE OUT LIFECYCLE FOR EACH sample_idx
    # ----------------------------------------
    #
    # We want a dictionary:
    #    lifecycle[sample_idx] = {
    #        "first_frame":    frame index where sample_idx first appears in top‐k,
    #        "fade_start":     frame index to begin fading alpha,
    #        "fade_end":       frame index where alpha=0 and hide_viewport = True,
    #        "is_persistent":  True if (sample_idx == global_best_idx at final frame),
    #    }
    lifecycle = {}
    #
    # For each Blender‐animation‐frame f (0‐based within this function),
    # we will figure out which “cumulative” count to use, then compute top‐k indices among scrs[:cumulative].
    # Then we record if a sample_idx appears in that top‐k. If it is the *first* time, record first_frame = f_global = start_frame + f.
    #
    frame_to_cumcount = []
    for i in range(N_RAND):
        frame_to_cumcount.append(rand_counts[i])
    for j in range(N_BO):
        frame_to_cumcount.append(bo_counts[j])

    # For each “animation frame index within this block,” collect top‐k:
    # Note: We will treat the very first frame of the animate_optimization cycle as “Blender frame = start_frame.”
    # So an internal loop‐index f in [0 .. total_frames-1] corresponds to Blender frame = (start_frame + f).
    for f in range(total_frames):
        cumcount = frame_to_cumcount[f]
        if cumcount <= 0:
            continue

        # If cumcount < k_best, then “top‐k” is simply the largest cumcount scores:
        # Equivalent to np.argsort(scrs[:cumcount])[-k_best:], but clipped by available.
        if f < N_RAND:
            best_here = np.arange(cumcount)[-k_best:]
        else:
            best_here = np.argsort(scrs[:cumcount])[-k_best:]
        best_here = set(best_here)

        blender_frame = start_frame + f

        for sample_idx in best_here:
            is_persist = ( (f == total_frames - 1) and (sample_idx == global_best_idx) )
            if sample_idx not in lifecycle:
                # This is the FIRST time sample_idx enters the top‐k at frame = blender_frame
                lifecycle[sample_idx] = {
                    "first_frame": blender_frame,
                    "last_frame" : blender_frame,
                    "is_persistent": is_persist
                }
            else:
                lifecycle[sample_idx] = {
                    "first_frame": lifecycle[sample_idx]["first_frame"],
                    "last_frame" : blender_frame,
                    "is_persistent": is_persist
                }

    # Now that every sample_idx that EVER appears in top‐k has a “last_frame,”
    # we can assign fade_end times. For non‐persistent points, fade from last → last + T_FADE.
    # For the one persistent point (global best at last frame), use a longer fade window (e.g. last+10 → last+40).
    for sample_idx, info in lifecycle.items():
        f0 = info["last_frame"]
        if info["is_persistent"]:
            fade_start = f0 + 40
            fade_end   = f0 + 50
        else:
            fade_start = f0
            fade_end   = f0 + T_FADE
        # Once alpha hits zero at fade_end, we hide it at fade_end + 1
        lifecycle[sample_idx]["fade_start"] = fade_start
        lifecycle[sample_idx]["fade_end"]   = fade_end

    # ----------------------------------------
    # 10.3 PASS 2: CREATE ONE CUBE PER sample_idx, SET UP MATERIAL + KEYFRAMES
    # ----------------------------------------
    #
    # 10.3.1— Create a “template” cube with bevel already applied, but hide it immediately.
    bpy.ops.mesh.primitive_cube_add(size=side_length)
    template_cube = bpy.context.active_object
    template_cube.name = "TemplateCube"

    # Add a bevel modifier (rounded corners) and apply it:
    bpy.ops.object.modifier_add(type='BEVEL')
    bev = template_cube.modifiers[-1]
    bev.width    = corner_radius
    bev.segments = 8
    bev.profile  = 0.5
    bpy.ops.object.modifier_apply(modifier=bev.name)

    # Assign a “base” material for all cubes (we will copy it per‐instance).
    template_metal_mat = bpy.data.materials.new("TemplateMetalMat")
    template_metal_mat.use_nodes = True
    bsdf_node = template_metal_mat.node_tree.nodes["Principled BSDF"]
    bsdf_node.inputs["Metallic"].default_value  = 0.5
    bsdf_node.inputs["Roughness"].default_value = 0.5
    #
    # IMPORTANT: Make sure the BSDF has an “Alpha” input exposed.
    bsdf_node.inputs["Alpha"].default_value = 1.0
    template_metal_mat.blend_method = 'BLEND'

    # Immediately hide the template so it doesn't render:
    template_cube.hide_viewport = True
    template_cube.hide_render   = True

    # 10.3.2— For each sample_idx that appeared at least once, create a copy of “TemplateCube”:
    for sample_idx, info in lifecycle.items():
        # Create a new Blender object by re‐using geometry from template_cube:
        new_name = f"SampleCube_{sample_idx}"
        new_obj = bpy.data.objects.new(name=new_name, object_data=template_cube.data.copy())
        bpy.context.collection.objects.link(new_obj)

        # Attach a copy of the material, so we can keyframe Alpha individually:
        mat_copy = template_metal_mat.copy()
        mat_copy.name = f"SampleMetal_{sample_idx}"
        new_obj.data.materials.clear()
        new_obj.data.materials.append(mat_copy)

        # Store offset location + orientation:
        x, y, heading = pts[sample_idx]
        x_world = x + offset[0]
        y_world = y + offset[1]
        z_world = default_z + (side_length/2)

        new_obj.location = (x_world, y_world, z_world)
        new_obj.rotation_euler = (0.0, 0.0, heading)
        bsdf_node_copy = mat_copy.node_tree.nodes["Principled BSDF"]
        norm_score = (scrs[sample_idx] / global_max) if (global_max > 0) else 0.0
        rgba = viridis_color(norm_score)
        bsdf_node_copy.inputs["Base Color"].default_value = rgba

        # Ensure the object is hidden initially (frame = 0 ... until first_frame - 1)
        new_obj.hide_viewport = True
        new_obj.hide_render   = True

        # Prepare animation data container:
        if new_obj.animation_data is None:
            new_obj.animation_data_create()
        if new_obj.animation_data.action is None:
            new_obj.animation_data.action = bpy.data.actions.new(name=f"Act_{new_name}")

        # Grab the BSDF’s Alpha socket so we can keyframe it:
        alpha_input = bsdf_node_copy.inputs["Alpha"]

        # 1) Keyframe “hidden” at frame = first_frame - 1:
        f_first = info["first_frame"]
        new_obj.hide_viewport = True
        new_obj.hide_render   = True
        new_obj.keyframe_insert(data_path="hide_viewport", frame=(f_first - 1))
        new_obj.keyframe_insert(data_path="hide_render",   frame=(f_first - 1))

        # 2) At frame = first_frame, show + set Alpha = 1.0:
        new_obj.hide_viewport = False
        new_obj.hide_render   = False
        new_obj.keyframe_insert(data_path="hide_viewport", frame=f_first)
        new_obj.keyframe_insert(data_path="hide_render",   frame=f_first)

        alpha_input.default_value = 0.5
        alpha_input.keyframe_insert(data_path="default_value", frame=f_first)

        # 3) At fade_start, begin fade (if not persistent, fade from 1 → 0 over T_FADE)
        f_fade_start = info["fade_start"]
        f_fade_end   = info["fade_end"]

        if not info["is_persistent"]:
            # Non‐persistent:    
            #   At fade_start, alpha = 1.0 (already keyframed if fade_start == first_frame)
            #   Insert keyframe alpha = 1.0 one frame before fade_start if fade_start > first_frame
            if f_fade_start > f_first:
                alpha_input.keyframe_insert(data_path="default_value", frame=f_fade_start)
            #   At fade_end, alpha = 0.0
            alpha_input.default_value = 0.0
            alpha_input.keyframe_insert(data_path="default_value", frame=f_fade_end)

            #   At (fade_end + 1), hide the cube entirely:
            new_obj.keyframe_insert(data_path="hide_viewport", frame=(f_fade_end))
            new_obj.keyframe_insert(data_path="hide_render",   frame=(f_fade_end))
            new_obj.hide_viewport = True
            new_obj.hide_render   = True
            new_obj.keyframe_insert(data_path="hide_viewport", frame=(f_fade_end + 1))
            new_obj.keyframe_insert(data_path="hide_render",   frame=(f_fade_end + 1))
        else:
            # Persistent (global best at last frame):
            #   We want a longer fade. By convention: fade from f_first+10 → f_first+40
            #   But we have already set alpha=1 at f_first. Insert alpha=1 at (f_first+9):
            fade_mid_start = f_fade_start
            fade_mid_end   = f_fade_end
            if fade_mid_start > f_first:
                alpha_input.default_value = 1.0
                alpha_input.keyframe_insert(data_path="default_value", frame=fade_mid_start - 10)
                alpha_input.keyframe_insert(data_path="default_value", frame=fade_mid_start)

            #   At fade_mid_end, alpha = 0.0:
            alpha_input.default_value = 0.0
            alpha_input.keyframe_insert(data_path="default_value", frame=fade_mid_end)

            #   At (fade_mid_end + 1), hide entirely:
            new_obj.keyframe_insert(data_path="hide_viewport", frame=(fade_mid_end))
            new_obj.keyframe_insert(data_path="hide_render",   frame=(fade_mid_end))
            new_obj.hide_viewport = True
            new_obj.hide_render   = True
            new_obj.keyframe_insert(data_path="hide_viewport", frame=(fade_mid_end + 1))
            new_obj.keyframe_insert(data_path="hide_render",   frame=(fade_mid_end + 1))

    # Clean up: Delete the “template” cube itself, since every real cube uses its mesh datablock already:
    bpy.data.objects.remove(template_cube, do_unlink=True)

    return total_frames


# ------------------------------------------------------------------------------------------------
# 11. DRAW NAVIGATION PATH
# ------------------------------------------------------------------------------------------------
def draw_navigation_path(base_pos_history, start_frame, offset, plot_frames=30):
    """
    • base_pos_history: array of shape (T,3). First two columns = (x,y).
    • Creates a solid‐orange emission tube along the path.
    • Animates bevel_factor_end from 0→1 over plot_frames frames.
    Returns: start_frame + plot_frames.
    """
    import bpy

    coords_2d = base_pos_history[:, :2]
    T = coords_2d.shape[0]

    # 1) Build a POLY curve whose control points are the (x,y) coords:
    curve_data = bpy.data.curves.new("NavPathCurve", type="CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new(type="POLY")
    spline.points.add(T - 1)

    for i, (x, y) in enumerate(coords_2d):
        z = 0.2  # raise it slightly above the ground
        spline.points[i].co = (x + offset[0], y + offset[1], z, 1.0)

    # Give the curve some thickness so it renders as a tube
    curve_data.bevel_depth = 0.05
    curve_data.bevel_resolution = 8

    curve_obj = bpy.data.objects.new("NavPath", curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # 2) Create a simple Emission material (solid orange)
    mat = bpy.data.materials.new(name="NavPathSolidMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in list(nodes):
        nodes.remove(node)

    # Create an Emission node and Material Output node
    node_emit   = nodes.new("ShaderNodeEmission")
    node_emit.location = (0, 0)
    target_srgb = (0.9176, 0.3451, 0.0471)
    def srgb_to_linear(c):
        return tuple(c_i ** 2.2 for c_i in c)
    linear_approx = srgb_to_linear(target_srgb)
    node_emit.inputs["Color"].default_value = (
        linear_approx[0],
        linear_approx[1],
        linear_approx[2],
        1.0
    )
    node_emit.inputs["Strength"].default_value = 1.0

    node_out    = nodes.new("ShaderNodeOutputMaterial")
    node_out.location = (200, 0)

    # Connect Emission → Surface
    links.new(node_emit.outputs["Emission"], node_out.inputs["Surface"])

    mat.blend_method = 'BLEND'
    curve_obj.data.materials.append(mat)

    # 3) Animate bevel_factor_end from 0→1 over [start_frame..start_frame+plot_frames]
    curve_data.bevel_factor_end = 0.0
    curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame)

    curve_data.bevel_factor_end = 1.0
    curve_data.keyframe_insert(data_path="bevel_factor_end", frame=start_frame + plot_frames)

    return start_frame + plot_frames

# ------------------------------------------------------------------------------------------------
# 12. ERASE NAVIGATION PATH BEHIND ROBOT (no USD re‐import)
# ------------------------------------------------------------------------------------------------
def animate_path_erasure(len_nav_steps, usd_start_frame):
    """
    Starting at Blender frame = usd_start_frame, animate NavPath.bevel_factor_start from 0→1
    over len_nav_steps. This “erases” the portion of the path behind the robot. Returns the
    last Blender frame used.
    """
    path_obj = bpy.data.objects.get("NavPath")
    if path_obj is None:
        raise RuntimeError("NavPath object not found for path erasure.")

    path_obj.data.bevel_factor_start = 0.0
    path_obj.data.keyframe_insert(data_path="bevel_factor_start", frame=usd_start_frame)

    path_obj.data.bevel_factor_start = 1.0
    path_obj.data.keyframe_insert(data_path="bevel_factor_start", frame=usd_start_frame + len_nav_steps - 1)

    # — now force linear interpolation —
    # 1) Make sure there *is* animation data on the curve_data
    curve_data = path_obj.data
    if curve_data.animation_data and curve_data.animation_data.action:
        # 2) Find the F-Curve for "bevel_factor_end"
        for fcu in curve_data.animation_data.action.fcurves:
            if fcu.data_path == "bevel_factor_start":
                # 3) Set every keyframe point in that F-Curve to LINEAR
                for kp in fcu.keyframe_points:
                    kp.interpolation = 'LINEAR'
                break

    return usd_start_frame + len_nav_steps - 1

# ------------------------------------------------------------------------------------------------
# 13. MAIN
# ------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    usd_subdir = args.usd_subdir.rstrip("/")
    output_dir = args.output_dir
    usd_start_frame = args.usd_start_frame
    os.makedirs(output_dir, exist_ok=True)
    bpy.context.preferences.edit.use_global_undo = False

    # 13.1. Parse metadata from subdir’s basename
    basename = os.path.basename(usd_subdir)
    env_name, layout_id, style_id, seed, ep_idx = parse_subdir_name(basename)
    print(f"Parsed: env={env_name}, layout={layout_id}, style={style_id}, seed={seed}, ep={ep_idx}")

    # 13.2. Find the single USD file (largest frame index)
    usd_frames_dir = os.path.join(usd_subdir, "usd", "frames")
    frame_files = glob(os.path.join(usd_frames_dir, "frame_*.usd"))
    if not frame_files:
        sys.exit(f"Error: No USD files found in {usd_frames_dir}")

    def frame_index(path):
        m = re.search(r'frame_(\d+)\.usd$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    frame_files_sorted = sorted(frame_files, key=frame_index)
    first_usd = frame_files_sorted[-1]  # pick the USD with the largest numeric index
    print(f"Using single USD file for import: {first_usd}")

    # 13.3. Set up Blender render & world
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_raytracing = True

    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    if world.use_nodes:
        bg = world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1)
            bg.inputs['Strength'].default_value = 1.0
    else:
        world.color = (0.0, 0.0, 0.0)

    ground = bpy.data.meshes.new("GroundPlane")
    ground_obj = bpy.data.objects.new("GroundPlane", ground)
    bpy.context.collection.objects.link(ground_obj)
    verts = [(-1000, -1000, -0.24), (1000, -1000, -0.24),
             (1000, 1000, -0.24), (-1000, 1000, -0.24)]
    faces = [(0, 1, 2, 3)]
    ground.from_pydata(verts, [], faces)
    ground.update()
    matte_mat = bpy.data.materials.new(name="MatteMaterial")
    matte_mat.use_nodes = True
    bsdf = matte_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.6
        bsdf.inputs["Metallic"].default_value = 0.2
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    ground_obj.data.materials.append(matte_mat)

    # Remove default objects
    for obj in list(bpy.data.objects):
        if obj.type in {'LIGHT', 'CAMERA'} or obj.name == 'Cube':
            bpy.data.objects.remove(obj, do_unlink=True)

    # 13.4. Import the single USD (at default time = 1)
    print_progress(0, 1, prefix="Importing room (time=1)")
    new_objs = import_single_usd(first_usd)
    print_progress(1, 1, prefix="Importing room (time=1)")

    # 13.5. Apply built‐in transform cache offset instead of a driver
    apply_transform_cache_offset(new_objs, usd_start_frame)

    apply_visibility_from_usd(new_objs, first_usd, usd_start_frame)

    # 13.6. Filter + retouch MESH objects
    remove_subs = [
        'finger1_collision','finger2_collision','finger1_pad_collision','finger2_pad_collision',
        'gripper0_right_ft_frame','gripper0_right_grip_site','gripper0_right_hand_collision',
        'mobilebase0_g0_col','mobilebase0_pedestal_feet_col','mobilebase0_g1_col','mobilebase0_support',
        'robot0_link0_collision','robot0_link1_collision','robot0_link2_collision',
        'robot0_link3_collision','robot0_link4_collision','robot0_link5_collision',
        'robot0_link6_collision','robot0_link7_collision'
    ]
    mesh_objs = [o for o in new_objs if o.type == 'MESH']
    kept = filter_and_retouch(mesh_objs, remove_subs)
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        if o not in kept:
            o.select_set(True)
    bpy.ops.object.delete()

    # 13.7. Compute floor offset to center the room at (0,0)
    offset, floor_obj, orig_center = compute_floor_offset(kept, x_tile=0.0, y_tile=0.0)

    # 13.8. Add area light above center
    lx, ly = (floor_obj.dimensions.x, floor_obj.dimensions.y)
    area_light_data = bpy.data.lights.new(name='AreaLight', type='AREA')
    area_light_data.energy = 400
    area_light_data.shape = 'RECTANGLE'
    area_light_data.size   = ly * 50
    area_light_data.size_y = lx * 50
    area_light = bpy.data.objects.new('AreaLight', area_light_data)
    bpy.context.collection.objects.link(area_light)
    area_light.location = (0, 0, 3.1)
    area_light.rotation_euler = (0, 0, 0)

    # 13.9. Parent Empty for the room
    parent = bpy.data.objects.new("RoomParent", None)
    bpy.context.collection.objects.link(parent)
    parent.matrix_world = Matrix.Translation(offset)
    for o in kept:
        o.parent = parent

    # 13.10. Create “center” Empty & reset camera (tracking→top-down)
    center_empty = create_center_empty(kept)
    dx_est = 14.0
    dy_est = 14.0
    cam = reset_camera(center_empty, dx_est, dy_est, T0=30)

    # 13.11. Animate BoT points using “OLD” NPZ
    method_name  = "mobipi"
    policy_name  = "bc_xfmr"
    old_data_dir = os.path.join(
        LOG_ROOT_DIR,
        env_name,
        method_name,
        policy_name,
        f"layout{layout_id}_style{style_id}_seed{seed}"
    )
    old_npz_path = os.path.join(old_data_dir, f"ep{ep_idx}_info.npz")
    if not os.path.isfile(old_npz_path):
        raise RuntimeError(f"Could not find OLD optimization file '{old_npz_path}'")

    anim_start = 50
    total_bo_frames = animate_optimization(old_npz_path, offset, start_frame=anim_start)
    bo_end_frame = anim_start + total_bo_frames - 1

    # 13.12. Draw navigation path using “NEW” NPZ (inside usd_subdir)
    new_npz_path = os.path.join(usd_subdir, f"ep{ep_idx}_info.npz")
    if not os.path.isfile(new_npz_path):
        raise RuntimeError(f"Could not find NEW optimization file '{new_npz_path}'")

    new_data = np.load(new_npz_path, allow_pickle=True)
    base_pos_history = new_data["base_pos_history"]

    path_draw_start = bo_end_frame + 10
    path_end_frame  = draw_navigation_path(base_pos_history, start_frame=path_draw_start, offset=offset)

    # 13.13. Zoom camera back over next 30 frames
    zoom_start = path_end_frame + 1
    zoom_end   = zoom_start + 30

    cam.keyframe_insert(data_path="location", frame=zoom_start - 1)
    cam.keyframe_insert(data_path="rotation_euler", frame=zoom_start - 1)

    dist = max(dx_est, dy_est)
    angle = math.radians(35)
    height = dist * math.tan(angle)
    track_loc = center_empty.location
    cam.location = (
        track_loc.x,
        track_loc.y - dist,
        height
    )
    mat_look = (center_empty.location - cam.location).to_track_quat('-Z','Y').to_euler()
    cam.rotation_euler = mat_look
    cam.keyframe_insert(data_path="location", frame=zoom_end)
    cam.keyframe_insert(data_path="rotation_euler", frame=zoom_end)

    # 13.14. Animate path erasure starting at usd_start_frame
    len_nav_steps = base_pos_history.shape[0]
    usd_final_frame = animate_path_erasure(len_nav_steps, usd_start_frame)

    # 13.15. Render all frames up to the maximum used
    tot_frames = usd_start_frame + frame_index(first_usd)

    if args.debug:
        blendfile = os.path.join(output_dir, 'debug_single_room.blend')
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_mainfile(filepath=blendfile)

    for f in range(1, tot_frames + 1):
        print_progress(f, tot_frames, prefix="Rendering")
        bpy.context.scene.frame_set(f)
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"frame_{f:04d}.png")
        bpy.ops.render.render(write_still=True)

    print("All done!")

if __name__ == "__main__":
    main()
