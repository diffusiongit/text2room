import os
import json
from PIL import Image

from model.text2room_pipeline import Text2RoomPipeline
from model.utils.opt import get_default_parser
from model.utils.utils import save_poisson_mesh, generate_first_image

import torch


@torch.no_grad()
def main(args):
    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        args.prompt = trajectories[0]["prompt"]

    # get first image from text prompt or saved image folder
    assert not args.input_image_path, "in layout-control we assume we have no start image, please do not specify --input_image_path"
    print("Generate front-facing image with prompt", args.prompt)
    first_image_pil = generate_first_image(args)

    # load pipeline
    pipeline = Text2RoomPipeline(args, first_image_pil=first_image_pil)

    # generate using all trajectories
    offset = 1  # have the start image already
    for t in trajectories:
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)

    # save outputs before completion
    pipeline.clean_mesh()
    intermediate_mesh_path = pipeline.save_mesh("after_generation.ply")
    save_poisson_mesh(intermediate_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # Now no longer need the models
    pipeline.remove_models()

    # save additional output
    pipeline.save_animations()
    pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
    pipeline.save_nerf_transforms()
    pipeline.save_seen_trajectory_renderings(apply_noise=True)

    print("Finished. Outputs stored in:", args.out_path)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)
