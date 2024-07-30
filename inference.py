import os
import imageio
import numpy as np
from PIL import Image
import torch
torch.manual_seed(1024)

from inference_utils import inference
from face_utils import get_face_img, get_faces_video
from batch_face import RetinaFace
face_detector = RetinaFace(gpu_id=0) if torch.cuda.is_available() else RetinaFace(gpu_id=-1)


def check_if_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"])


def check_if_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4", ".avi"])

def concat_image(image1, image2, image3):
    # resize to the same size of image3
    image1     = image1.resize(image3.size)
    image2     = image2.resize(image3.size)
    concat_img = Image.new("RGB", (image3.width*3, image3.height))
    concat_img.paste(image1, (0, 0))
    concat_img.paste(image2, (image3.width, 0))
    concat_img.paste(image3, (image3.width*2, 0))
    return concat_img

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--id_input", type=str, help="Path to the input, can be an image, a video", required=True)
    parser.add_argument("--makeup_reference", type=str, help="Path to the makeup image file", required=True)
    parser.add_argument("--fast_test", action="store_true", help="Use fast test mode, only process every 5 frames")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    id_input         = args.id_input
    makeup_reference = args.makeup_reference
    output_dir       = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # check if the input is a video or an image
    id_basename = os.path.basename(id_input).split(".")[0]
    if check_if_video_file(id_input):
        # read all frames from the video
        frames, coords = get_faces_video(face_detector, id_input)
        id_images = frames if not args.fast_test else frames[::5]
        coords = coords if not args.fast_test else coords[::5]
    elif check_if_image_file(id_input):
        frame, coord = get_face_img(face_detector, id_input)
        id_images = [frame]
        coords = [coord]
    else:
        raise ValueError("Unsupported file format for id_input")

    makeup_basename = os.path.basename(makeup_reference).split(".")[0]
    if check_if_image_file(makeup_reference):
        makeup_image_pil, _ = get_face_img(face_detector, makeup_reference)
    else:
        raise ValueError("Unsupported file format for makeup_reference")

    if len(id_images) == 0:
        raise ValueError("No input images loaded")
    elif len(id_images) == 1:
        result_img = inference(id_images[0], makeup_image_pil)
        # concat id, makeup and result images
        concat_img = concat_image(id_images[0], makeup_image_pil, result_img)
        concat_img.save(os.path.join(output_dir, id_basename + makeup_basename + '.png'))
        print(f"Output Image Saved to {os.path.join(output_dir, id_basename + makeup_basename + '.png')}")
    elif len(id_images) > 1:
        # get fps of the original video
        try:
            fps = imageio.get_reader(id_input).get_meta_data()["fps"]
        except:
            print("Failed to get the fps of the video, using default 25 fps")
            fps = 25
        writer = imageio.get_writer(os.path.join(output_dir, id_basename + makeup_basename + '.mp4'), fps=fps if not args.fast_test else fps/5, quality=9, codec="libx264")
        for id_image_pil in tqdm(id_images):
            result_img = inference(id_image_pil, makeup_image_pil)
            concat_img = concat_image(id_image_pil, makeup_image_pil, result_img)
            writer.append_data(np.array(concat_img))
        writer.close()
        print(f"Output Video Saved to {os.path.join(output_dir, id_basename + makeup_basename + '.mp4')}")
