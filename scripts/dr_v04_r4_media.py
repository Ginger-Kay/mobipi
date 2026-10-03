"""Bind six native movies to existing receipts and generate labeled copies."""
import argparse,csv,json,hashlib,html,subprocess
from pathlib import Path
from datetime import datetime,timezone
import cv2
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def movie_info(p,decode=False):
    cap=cv2.VideoCapture(str(p));assert cap.isOpened()
    frames=0;fps=cap.get(cv2.CAP_PROP_FPS);width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if decode:
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(int(height),int(width),3);frames+=1
    else:frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release();return dict(frames=frames,fps=fps,width=int(width),height=int(height))
def main():
    a=argparse.ArgumentParser();a.add_argument("--run",type=Path,required=True);args=a.parse_args();r=args.run
    roster=load(r/"frozen-six/roster.json");rows=[];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for slot in roster["slots"]:
        root=r/"episodes"/f'{slot["slot"]:02d}-{slot["group_id"]}'
        files=list(root.glob("engineering-attempt-*/completed.json"));assert len(files)==1
        complete=load(files[0]);attempt=Path(complete["attempt"]);n=complete["result"]["steps"]
        native=load(attempt/"task-video-manifest.json");panobinding=load(attempt/"panoramic-binding.json")
        assert native["binding"]["run_id"]==r.name and native["binding"]["group_id"]==slot["group_id"]
        assert native["binding"]["route"]==complete["route"] and native["steps"]==n
        assert len(native["decoded_frames_sha256"])==n
        assert sha(attempt/"original.mp4")==native["files"]["original.mp4"]["sha256"]
        assert sha(attempt/"panoramic.mp4")==panobinding["sha256"]
        info=movie_info(attempt/"panoramic.mp4",decode=True);assert info["frames"]==n and info["fps"]==20
        out=r/"videos"/f'{slot["slot"]:02d}-{slot["group_id"]}';out.mkdir(exist_ok=True)
        watch=out/"watch-labeled.mp4";label=out/"label.txt"
        label.write_text(f'{slot["group_id"]} | MODEL SELECTED {complete["route"]} | step2000 | {complete["result"]["reason"]} | native video 20Hz')
        # This packaged ffmpeg has overlay but no drawtext. Render exact native
        # timestamps into transparent PNG frames using Pillow instead.
        labels=out/"label-frames";labels.mkdir(exist_ok=False)
        font=ImageFont.truetype("/share/personal/chensiyu/haokaijiang/MobiWAM/env/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",23)
        trace=[json.loads(s) for s in (attempt/"trace.jsonl").read_text().splitlines()]
        samecamera=native["binding"]["camera"]==panobinding["camera"]
        for i,t in enumerate(trace):
            banner=Image.new("RGBA",(1920,78),(0,0,0,170));draw=ImageDraw.Draw(banner)
            draw.text((12,7),label.read_text(),font=font,fill="white")
            draw.text((12,40),f'frame {i} | native sim time {t["after"]["sim_time"]:.3f}s | video time {(i+1)/20:.3f}s | base inset: '+("same camera" if samecamera else "wider simultaneous camera"),font=font,fill="white")
            banner.save(labels/f"label-{i:05d}.png")
        vf="[1:v]scale=480:270[base];[0:v][base]overlay=W-w-12:H-h-12[main];[main][2:v]overlay=0:0:shortest=1[out]"
        cmd=[ffmpeg,"-hide_banner","-loglevel","error","-threads","2","-i",str(attempt/"original.mp4"),"-i",str(attempt/"panoramic.mp4"),
            "-framerate","20","-i",str(labels/"label-%05d.png"),
            "-filter_complex_threads","1","-filter_complex",vf,"-map","[out]","-an","-c:v","libx264","-threads","2","-preset","veryfast","-crf","20","-movflags","+faststart",str(watch)]
        process=subprocess.run(cmd,capture_output=True,text=True)
        if process.returncode:
            write(out/"transcode-failure.json",dict(command=cmd,returncode=process.returncode,stderr=process.stderr))
            raise RuntimeError(process.stderr)
        wi=movie_info(watch,decode=True);assert wi["frames"]==n
        cap=cv2.VideoCapture(str(attempt/"original.mp4"));sheet=Image.new("RGB",(1440,320),"black");draw=ImageDraw.Draw(sheet)
        for k,frame_index in enumerate((0,n//2,n-1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read();assert ok
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));image.thumbnail((480,270));sheet.paste(image,(480*k,35))
            draw.text((480*k+4,8),f'{complete["route"]} frame {frame_index}, t={(frame_index+1)/20:.2f}s',fill="white")
        cap.release();sheet.save(out/"preview.jpg")
        terminal=load(files[0].parent/"terminal-live-state.json")
        primary_binding=load(files[0].parent/"process.json")
        new=dict(slot=slot["slot"],group_id=slot["group_id"],task=slot["scientific_row"]["task"],route=complete["route"],
             result=complete["result"]["reason"],checker_success=complete["result"]["checker_success"],frames=n,video_duration_s=n/20,
             original_video=str(attempt/"original.mp4"),original_sha256=native["files"]["original.mp4"]["sha256"],watch_video=str(watch),watch_sha256=sha(watch),
             panoramic_video=str(attempt/"panoramic.mp4"),panoramic_sha256=panobinding["sha256"],preview=str(out/"preview.jpg"),
             native_manifest=str(attempt/"task-video-manifest.json"),native_target=native["binding"]["native"]["target_description"],
             checkpoint_sha256=roster["checkpoint_sha256"],episode_code_commit=primary_binding["code_commit"],full_recorded_control_steps=True,
             partial_terminal_control_step=terminal["partial_control_step"],base_view_distinct_camera=not samecamera,native_video_all_frames_decoded_receipt_reused=True,
             new_panoramic_and_watch_all_frames_decoded=True,full_action_replay=False,development_repeat=True,formal_train_ready=False)
        write(out/"episode-video-manifest.json",new);rows.append(new)
        print(json.dumps(dict(slot=slot["slot"],frames=n,watch=str(watch))),flush=True)
    with (r/"videos/videos.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md="# R4 six new native feedback episodes\n\nDevelopment repeats; cached preoutcome model input; no new independent Sources/test.\n\n|slot|Source|route|result|primary video absolute path|watching copy|\n|---|---|---|---|---|---|\n"
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>DR-v0.4-R4 six simulator episodes</title><style>body{max-width:1200px;margin:auto;font:16px sans-serif;background:#111;color:#eee}video{width:100%}a{color:#87c5ff}section{margin:2em 0}img{max-width:100%}</style><h1>R4模型选路后的六次新仿真</h1><p>固定step2000、原Source恢复、缓存采前输入、真实反馈控制。开发重复执行；旧正式结果不覆盖。</p>'
    for x in rows:
        md+=f'|{x["slot"]}|{x["group_id"]}|{x["route"]}|{x["result"]}|{x["original_video"]}|{x["watch_video"]}|\n'
        relative=Path(x["watch_video"]).relative_to(r/"videos");preview=Path(x["preview"]).relative_to(r/"videos")
        page+=f'<section><h2>{x["slot"]}. {html.escape(x["group_id"])} · {x["route"]}</h2><p>{html.escape(x["native_target"])}<br>{x["result"]}; {x["frames"]} frames; full replay not performed.</p><video controls preload="metadata" src="{relative}" poster="{preview}"></video><p>主原片绝对路径：<code>{html.escape(x["original_video"])}</code><br>SHA：<code>{x["original_sha256"]}</code></p><img src="{preview}" alt="first middle final native frames"></section>'
    (r/"videos/videos.md").write_text(md);(r/"videos/index.html").write_text(page+"</html>")
    write(r/"videos/completion.json",dict(ended_at=datetime.now(timezone.utc).isoformat(),slots=6,primary_videos=6,watch_copies=6,panoramic_videos=6,
        native_original_integrity=True,full_frame_sync=True,summary=rows,training_optimizer_steps=0,test_read=False))
if __name__=="__main__":main()
