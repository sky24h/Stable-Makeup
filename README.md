# A more clean, easy to use version

The environment setup and code structure of the original repo are just terrible.

So , I forked it to create a cleaner version.

### Update
- [x] Created a Gradio demo on Hugging Face: https://huggingface.co/spaces/sky24h/Stable-Makeup-unofficial

## Environment Setup
Run the following commands to install the required packages:
```shell
pip install -r requirements.txt
```

If you encounter CUDA verision mismatches error, try to install torch manually to specify the version that matches your cuda version.
For example, if you have CUDA 11.8, you can install torch with the following command:
```shell
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

You can find the correct version of torch here: https://pytorch.org/get-started/previous-versions/

## Inference
To inference, just run the following command:
```shell
python inference.py --id_input ./test_imgs/id/1.jpg --makeup_reference ./test_imgs/makeup/1.jpg --output_dir ./output
```

id_input can be an image or a video, and makeup_reference should be an image.

Original README:
----------------
# Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model

<a href="https://arxiv.org/abs/2403.07764"><img src="https://img.shields.io/badge/arXiv-2403.07764-b31b1b.svg" height=22.5></a>
![teaser](assets/sm_teaser.jpg)
Our proposed framework, Stable-Makeup, is a novel diffusion-based method for makeup transfer that can robustly transfer a diverse range of real-world makeup styles, from light to extremely heavy makeup.

## Method Details
![method](https://github.com/Xiaojiu-z/Stable-Makeup/blob/main/assets/sm_method.jpg)
Given a source image $\mathit{I_s}$ , a reference makeup image $\mathit{I_m}$ and an obtained facial structure control image $\mathit{I_c}$ , Stable-Makeup utilizes D-P makeup encoder to encode $\mathit{I_m}$. Content and structural encoders are used to encode $\mathit{I_s}$ and $\mathit{I_c}$ respectively. With the aid of the makeup cross-attention layers, Stable-Makeup aligns the facial regions of $\mathit{I_s}$ and $\mathit{I_m}$ , enabling successful transfers the intricate makeup details. After content-structure decoupling training, Stable-Makeup further maintains content and structure of $\mathit{I_s}$ .

## Todo List
1. - [x] inference and training code
2. - [x] pre-trained weights

## Getting Started
### Environment Setup
Our code is built on the [diffusers](https://github.com/huggingface/diffusers/) version of Stable Diffusion v1-5. We use [SPIGA](https://github.com/andresprados/SPIGA) and [facelib](https://github.com/sajjjadayobi/FaceLib) to draw face structural images. 
```shell
git clone https://github.com/Xiaojiu-z/Stable-Makeup.git
cd Stable-Makeup
```
### Pretrained Models
[Google Drive](https://drive.google.com/drive/folders/1397t27GrUyLPnj17qVpKWGwg93EcaFfg?usp=sharing).
Download them and save them to the directory `models/stablemakeup`. One deviation from the original paper is randomly dropping out inputs into the structural encoder during training, resulting in improved semantic alignment. Enjoy it!

### Inference

```python
python infer_kps.py
```

### Training
You can prepare datasets following our paper and make a jsonl file (each line with 4 key-value pairs, including original id, edited id, augmented id, face structural image of edited id) or you can implement a dataset and a dataloader class by yourself (Probably faster than organizing into my data form).

```python
bash train.sh
```

### Gradio demo
We provide a simple gr demo for more flexible use.
```python
python gradio_demo_kps.py
```

## Citation
```
@article{zhang2024stable,
  title={Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model},
  author={Zhang, Yuxuan and Wei, Lifu and Zhang, Qing and Song, Yiren and Liu, Jiaming and Li, Huaxia and Tang, Xu and Hu, Yao and Zhao, Haibo},
  journal={arXiv preprint arXiv:2403.07764},
  year={2024}
}
```
