import os
import torch
import pkg_resources
from PIL import Image
from facelib import FaceDetector
from spiga_draw import spiga_process, spiga_segmentation
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework

from gdown import download_folder
from gdown import download as gdown_download

from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from detail_encoder.encoder_plus import detail_encoder
from diffusers.utils import load_image

processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")


def get_draw(pil_img, size):
    spigas = spiga_process(pil_img, detector)
    if spigas == False:
        width, height = pil_img.size
        black_image_pil = Image.new("RGB", (width, height), color=(0, 0, 0))
        return black_image_pil
    else:
        spigas_faces = spiga_segmentation(spigas, size=size)
        return spigas_faces


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"])


def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)


def init_pipeline():
    # Initialize the model
    model_id = "runwayml/stable-diffusion-v1-5"  # or your local sdv1-5 path
    base_path = "./checkpoints/stablemakeup"
    folder_id = "1397t27GrUyLPnj17qVpKWGwg93EcaFfg"
    if not os.path.exists(base_path):
        download_folder(id=folder_id, output=base_path, quiet=False, use_cookies=False)
    makeup_encoder_path = base_path + "/pytorch_model.bin"
    id_encoder_path = base_path + "/pytorch_model_1.bin"
    pose_encoder_path = base_path + "/pytorch_model_2.bin"

    Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
    id_encoder = ControlNetModel.from_unet(Unet)
    pose_encoder = ControlNetModel.from_unet(Unet)
    makeup_encoder = detail_encoder(Unet, "openai/clip-vit-large-patch14", "cuda", dtype=torch.float32)
    makeup_state_dict = torch.load(makeup_encoder_path)
    id_state_dict = torch.load(id_encoder_path)
    id_encoder.load_state_dict(id_state_dict, strict=False)
    pose_state_dict = torch.load(pose_encoder_path)
    pose_encoder.load_state_dict(pose_state_dict, strict=False)
    makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
    id_encoder.to("cuda")
    pose_encoder.to("cuda")
    makeup_encoder.to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, safety_checker=None, unet=Unet, controlnet=[id_encoder, pose_encoder], torch_dtype=torch.float32
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe, makeup_encoder


def inference(pipeline, makeup_encoder, id_image_path, makeup_image_path):
    id_image = load_image(id_image_path).resize((512, 512))
    makeup_image = load_image(makeup_image_path).resize((512, 512))
    pose_image = get_draw(id_image, size=512)
    result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, pipe=pipeline, guidance_scale=1.6)
    return result_img


if __name__ == "__main__":
    import glob
    from tqdm import tqdm
    from natsort import natsorted
    torch.manual_seed(1024)

    pipeline, makeup_encoder = init_pipeline()
    id_folder = "./test_imgs/input_video"
    makeup_folder = "./test_imgs/makeup"
    out_folder = "./output"
    all_id_images = natsorted(glob.glob(os.path.join(id_folder, "*.png")))
    all_makeup_images = natsorted(glob.glob(os.path.join(makeup_folder, "*.jpg")))

    for id_image_path in tqdm(all_id_images):
        for makeup_image_path in all_makeup_images:
            result_img = inference(pipeline, makeup_encoder, id_image_path, makeup_image_path)
            result_img.save(os.path.join(out_folder, os.path.basename(id_image_path) + "_" + os.path.basename(makeup_image_path)))
