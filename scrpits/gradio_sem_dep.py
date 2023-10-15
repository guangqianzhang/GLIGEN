import argparse
import json
import os
import sys
from functools import partial
import torch
import numpy as np


import gradio as gr
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from tqdm import trange

from gligen_inference import load_ckpt, crop_and_resize, alpha_generator, set_alpha_scale
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from scrpits.utils import prepare_batch_sem_depth

device='cuda:0'
torch.set_grad_enabled(False)


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    print(f'w:{pad_w} h:{pad_h}')
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    print(f'image shape:{im_padded.shape}')
    return im_padded


@torch.no_grad()
def predict(image,dep,sem, prompt,negative_prompt,
            sampler_type,steps, num_samples, scale,
            seed, eta, strength,alpha):

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - prepare batch - - - - - #
    # if "keypoint" in meta["ckpt"]:
    starting_noise = torch.randn(num_samples, 4, 64, 64).to(device)

    seed_everything(seed)
    batch = prepare_batch_sem_depth(image,dep, sem,prompt,num_samples)
    context = text_encoder.encode(  [prompt]*num_samples  )
    uc = text_encoder.encode( num_samples*[""] )
    if negative_prompt is not None:
        uc = text_encoder.encode( num_samples*[negative_prompt] )


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process

        # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    # - - - - - sampler - - - - - #
    alpha_type=None
    a=(1-alpha)*2.3
    # alpha_type=[a, 1-a-alpha, alpha]

    alpha_generator_func = partial(alpha_generator, type=alpha_type)
    if sampler_type==0:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        # steps = 250
    elif sampler_type==1:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)


        # steps = 50
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    starting_noise = torch.randn(num_samples, 4, 64, 64).to(device)
    z = autoencoder.encode(batch['image'] )  # 1x4x122x200  wh/8
    z_enc = sampler.stochastic_encode(
            z, torch.tensor([t_enc] * num_samples).to(device))  # 1x4x122x200

    input = dict(
        x=z_enc,
        timesteps=None,
        context=context,
        grounding_input=grounding_input,
        inpainting_extra_input=None,
        grounding_extra_input=grounding_extra_input)
    # - - - - - start sampling - - - - - #
    shape = (num_samples, model.in_channels, model.image_size, model.image_size)

    with trange(n_iter,desc="Sampling") as t:
        for i in t:
            samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=scale,ddim_eta=eta)
            # samples_fake = sampler.decode(input=input, t_start=t_enc,shape=shape, guidance_scale=scale,uc=uc, )  # 2x4x112x200
            samples_fake = autoencoder.decode(samples_fake).cpu()

            samples = torch.clamp(samples_fake, min=-1, max=1) * 0.5 + 0.5
            torchvision.utils.save_image(samples, 'out_image_tensor.png', nrow=8, normalize=True,
                                         scale_each=True, range=(-1, 1))
            print(f'sample shape:{samples.shape}')
            samples = torch.nn.functional.interpolate(
                samples,
                size=(900, 1600),
                mode="bicubic",
                align_corners=False,
            )
            samples = samples.cpu().numpy().transpose(0, 2, 3, 1) * 255
            t.update(1)
            print(f'sample shape:{samples.shape}')
            # sample = mage.fromarray(sample.astype(np.uint8))
            # sample.show()

    return [Image.fromarray(img.astype(np.uint8)) for img in samples]


print('gradio sem depth')

# ckpt='/home/cqjtu/GLIGEN/OUTPUT/test/tag00/checkpoint_00415001.pth'
# ckpt='/home/cqjtu/GLIGEN/OUTPUT/test/tag01/checkpoint_00480001.pth'

# run(meta, args, starting_noise)
# - - - - - prepare models - - - - - #
model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt)
# prompt='a parking lot filled with lots of parked red cars'
negative_prompt='nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, ' \
                'worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet,'

file='n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984237447423'
root='/home/cqjtu/Documents/dataset/v1.0-mini'
camer="/CAM_BACK_LEFT"
dep=root+f'/samples_{camer}/dep/{file}.png'
sem=root+f'/samples_{camer}/seg/{file}.png'
image=root+f'/samples{camer}/{file}.jpg'
# image=Image.open(image)
with open(root+f"/samples_{camer}/new_caption.json",'r') as f:
    data=json.load(f)
    prompt=data[f'{file}.jpg']
    print(prompt)
prompt=prompt+''
sampler=0
ddim_steps=20
num_samples=1
scale=5
seed=623418984
eta=0
strength=0.99
alpha=0.7
n_iter=1
gallery=predict(  image, dep,sem, prompt, negative_prompt,
                    sampler, ddim_steps, num_samples, scale,
                    seed, eta, strength,alpha)
for img in gallery:
    img.show()
# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Stable Diffusion GLIGEN2Img")
#
#     with gr.Row():
#         with gr.Column():
#             dep = gr.Image(label='dep',source='upload', type="filepath")
#             sem = gr.Image(label='sem',source='upload', type="filepath")
#             prompt = gr.Textbox(label="Prompt")
#             negative_prompt=gr.Textbox(label="negative_prompt")
#             run_button = gr.Button(label="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(
#                     label="Images", minimum=1, maximum=4, value=1, step=1)
#                 sampler = gr.Slider(label="sampler 0:DDIM 1:PLMS", minimum=0,  maximum=1, value=0, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=150, step=1)
#                 scale = gr.Slider(
#                     label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1
#                 )
#                 strength = gr.Slider(
#                     label="Strength", minimum=0.0, maximum=1.0, value=0.9, step=0.01
#                 )
#                 seed = gr.Slider(
#                     label="Seed",
#                     minimum=0,
#                     maximum=2147483647,
#                     step=1,
#                     randomize=True,
#                 )
#                 alpha= gr.Number(label="alpha ", value=0.7)
#                 eta = gr.Number(label="eta (DDIM)", value=0.0)
#         with gr.Column():
#             image = gr.Image(label='image',source='upload', type="pil")
#             gallery = gr.Gallery(label="Generated images", show_label=False).style(
#                 grid=[num_samples], height="auto",margin="auto")
#
#     run_button.click(fn=predict, inputs=[
#                      image, dep,sem, prompt, negative_prompt,
#                     sampler, ddim_steps, num_samples, scale,
#                     seed, eta, strength,alpha], outputs=[gallery])
#
#
# block.launch(share=True)
