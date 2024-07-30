import gradio as gr
from inference_utils import inference

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Interface(
            fn=lambda id_image, makeup_image, guidance_scale=1.6: inference(id_image, makeup_image, guidance_scale=guidance_scale, size=512),
            inputs=[
                gr.Image(type="pil", label="id_image", height=512, width=512),
                gr.Image(type="pil", label="makeup_image", height=512, width=512),
                gr.Slider(minimum=1.01, maximum=3, value=1.6, step=0.05, label="guidance_scale", info="1.05-1.15 is suggested for light makeup and 2 for heavy makeup."),
            ],
            outputs="image",
            title="Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model",
            allow_flagging="never",
            examples=[
                ["./test_imgs/id/1.jpg", "./test_imgs/makeup/1.jpg"],
                ["./test_imgs/id/3.jpg", "./test_imgs/makeup/3.jpg"],
            ],
            cache_examples=True,
        )
        demo.queue().launch()
