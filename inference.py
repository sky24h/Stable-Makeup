import os
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
torch.manual_seed(1024)

from inference_utils import init_pipeline, inference
pipeline = init_pipeline()


def read_video_frames_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def check_if_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"])


def check_if_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4", ".avi"])


if __name__ == "__main__":
    import glob
    from tqdm import tqdm
    from natsort import natsorted

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--id_input", type=str, help="Path to the input, can be an image, a video", required=True)
    parser.add_argument("--makeup_reference", type=str, help="Path to the makeup image file", required=True)
    parser.add_argument("--out_folder", type=str, default="./output")
    args = parser.parse_args()

    id_input         = args.id_input
    makeup_reference = args.makeup_reference
    out_folder       = args.out_folder
    os.makedirs(out_folder, exist_ok=True)

    pipeline, makeup_encoder = init_pipeline()

    # check if the input is a video or an image
    id_basename = os.path.basename(id_input).split(".")[0]
    if check_if_video_file(id_input):
        # read all frames from the video
        id_images = read_video_frames_pil(id_input)
    elif check_if_image_file(id_input):
        id_images = [Image.open(id_input)]
    else:
        raise ValueError("Unsupported file format for id_input")

    if check_if_image_file(makeup_reference):
        makeup_image_pil = Image.open(makeup_reference)
    else:
        raise ValueError("Unsupported file format for makeup_reference")

    if len(id_images) == 0:
        raise ValueError("No input images loaded")
    elif len(id_images) == 1:
        result_img = inference(pipeline, makeup_encoder, id_images[0], makeup_image_pil)
        result_img.save(os.path.join(out_folder, id_basename + "_makeup.png"))
        print(f"Output Image Saved to {os.path.join(out_folder, id_basename + '_makeup.png')}")
    elif len(id_images) > 1:
        writer = imageio.get_writer(os.path.join(out_folder, id_basename + "_makeup.mp4"), fps=10, quality=9, codec="libx264")
        for id_image_pil in tqdm(id_images):
            result_img = inference(pipeline, makeup_encoder, id_image_pil, makeup_image_pil)
            writer.append_data(np.array(result_img))
        writer.close()
        print(f"Output Video Saved to {os.path.join(out_folder, id_basename + '_makeup.mp4')}")
