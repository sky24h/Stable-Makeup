import os
import gradio as gr
import torch

from inference_utils import init_pipeline

pipeline = init_pipeline()
torch.cuda.set_device(0)


# Define your ML model or function here
def model_call(id_image, makeup_image, num):
    # # Your ML logic goes here
    id_image = Image.fromarray(id_image.astype("uint8"), "RGB")
    makeup_image = Image.fromarray(makeup_image.astype("uint8"), "RGB")
    id_image = id_image.resize((512, 512))
    makeup_image = makeup_image.resize((512, 512))
    pose_image = get_draw(id_image, size=512)
    result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, guidance_scale=num, pipe=pipeline)
    return result_img


# Create a Gradio interface
image1 = gr.inputs.Image(label="id_image")
image2 = gr.inputs.Image(label="makeup_image")
number = gr.inputs.Slider(minimum=1.01, maximum=5, default=1.5, label="makeup_guidance_scale")
output = gr.outputs.Image(type="pil", label="Output Image")

iface = gr.Interface(
    fn=lambda id_image, makeup_image, num: model_call(id_image, makeup_image, num),
    inputs=[image1, image2, number],
    outputs=output,
    title="Facial Makeup Transfer Demo",
    description="Upload 2 images to see the model output. 1.05-1.15 is suggested for light makeup and 2 for heavy makeup",
)
# Launch the Gradio interface
iface.queue().launch(server_name="0.0.0.0")
