"""
A script to run collect_images.py in batch.

Usage:
    python collect_images_batch.py --env_names [comma-separated env names] \
        --seeds [comma-separated list of seeds] \
        --style_ids [comma-separated list of style IDs] \
        --layout_ids [comma-separated list of layout IDs]

@yjy0625
"""
import os
import click

@click.command()
@click.option('--env_names', '-e', type=str, required=True, help="Comma-separated list of environment names.")
@click.option('--seeds', '-s', type=str, default="0", help="Comma-separated list of seeds.")
@click.option('--style_ids', '-i', type=str, required=True, help="Comma-separated list of layout IDs.")
@click.option('--layout_ids', '-i', type=str, default=None, help="Comma-separated list of layout IDs.")
def batch_collect_images(env_names, seeds, style_ids, layout_ids):
    """Run collect_images.py with combinations of env_names, seeds, and style/layout IDs."""
    env_names = env_names.split(",")
    seeds = [int(seed) for seed in seeds.split(",")]
    style_ids = [int(style_id) for style_id in style_ids.split(",")]
    if layout_ids is None:
        layout_ids = style_ids
    else:
        layout_ids = [int(layout_id) for layout_id in layout_ids.split(",")]

    for env_name in env_names:
        for seed in seeds:
            for style_id in style_ids:
                for layout_id in layout_ids:
                    command = (
                        f"python collect_images.py --env_name \"{env_name}\" "
                        f"--seed {seed} --layout_id {layout_id} --style_id {style_id}"
                    )
                    print(f"Running command: {command}")
                    os.system(command)

if __name__ == '__main__':
    batch_collect_images()
