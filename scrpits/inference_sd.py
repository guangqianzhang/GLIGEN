import os.path
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import gradio as gr
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from utils import load_ckpt, prepare_batch_sem_depth, alpha_generator, set_alpha_scale, inference_data_load
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
from trainer import batch_to_device


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


dep = '/home/cqjtu/Documents/dataset/gligen/test/dep/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.png'
sem = '/home/cqjtu/Documents/dataset/gligen/test/seg/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.png'
image = '/home/cqjtu/Documents/dataset/gligen/test/img/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg'
image = Image.open(image)


@torch.no_grad()
def inference(prompt, negative_prompt,
              sampler_type, ddim_steps, num_samples, scale,
              seed, eta, strength, alpha):
    model_wo_wrapper = model  # ?
    disable_inference_in_training = False
    batch_size = num_samples
    batch_here = batch_size
    # batch = prepare_batch_sem_depth(image, dep, sem, prompt, batch_size)  # 单一路径
    dataset = inference_data_load(image_rootdir, depth_rootdir, sem_rootdir, caption_path)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    starting_noise = torch.randn(batch_size, 4, 64, 64).to(device)
    # z = autoencoder.encode(batch["image"])  # 1x4x112x200
    # noise = torch.randn_like(z)
    seed_everything(seed)
    for data in tqdm(dataloader):
        batch = batch_to_device(data, device)
        # Do an inference on one training batch
        # keypoint case
        real_images_with_box_drawing = batch["image"] * 0.5 + 0.5  # 真实图片

        uc = text_encoder.encode(batch_here * [negative_prompt])
        context = text_encoder.encode(batch_here * ["".join(batch['caption'])])
        alpha_type = None
        alpha_generator_func = partial(alpha_generator, type=alpha_type)
        if sampler_type == 1:
            sampler = PLMSSampler(diffusion, model_wo_wrapper, alpha_generator_func=alpha_generator_func,
                                  set_alpha_scale=set_alpha_scale)
        elif sampler_type == 0:
            sampler = DDIMSampler(diffusion, model_wo_wrapper)
        shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)

        # extra input for inpainting
        inpainting_extra_input = None

        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

        grounding_extra_input = None
        if grounding_downsampler_input != None:
            grounding_extra_input = grounding_downsampler_input.prepare(batch)

        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
        model.grounding_tokenizer_input = grounding_tokenizer_input

        grounding_input = grounding_tokenizer_input.prepare(batch)
        input = dict(x=starting_noise,
                     timesteps=None,
                     context=context,
                     inpainting_extra_input=inpainting_extra_input,
                     grounding_extra_input=grounding_extra_input,
                     grounding_input=grounding_input)
        samples = sampler.sample(S=ddim_steps, shape=shape, input=input, uc=uc, guidance_scale=scale, ddim_eta=eta)

        autoencoder_wo_wrapper = autoencoder  # Note itself is without wrapper since we do not train that.
        samples = autoencoder_wo_wrapper.decode(samples).cpu()
        samples = torch.clamp(samples, min=-1, max=1) * 0.5 + 0.5
        if test_show:
            prom = ''
            prom.join(prompt.split(',')[1:])
            torchvision.utils.save_image(samples,
                                         f'{sampler_type}/{prompt}_{negative_prompt}_{ddim_steps}_{scale}_{seed}.png',
                                         nrow=8, normalize=True,
                                         scale_each=True, range=(-1, 1))

        samples = torch.nn.functional.interpolate(
            samples,
            size=(900, 1600),
            mode="bicubic",
            align_corners=False,
        )
        samples = samples.cpu().numpy().transpose(0, 2, 3, 1) * 255
        # print(f'sample shape:{samples.shape}')
        images = [Image.fromarray(sample.astype(np.uint8)) for sample in samples]
        imgs = []
        for indx, img in enumerate(images):
            # img.resize((900,1600),resample=3)
            img.save(os.path.join(save_path, batch['name'][indx].split('/')[-1]).replace('.jpg','.png'))
            imgs.append(img)
        if test_show:
            img = Image.fromarray(np.concatenate(imgs, axis=0))
            img.show()
        if use_gradio:
            return imgs
        # image_caption_saver(samples, real_images_with_box_drawing, None, batch["caption"],iter_name)


print('inference extra-gligen')
device = 'cuda:0'
# ckpt='/home/cqjtu/GLIGEN/OUTPUT/test/tag01/checkpoint_00485001.pth'
# ckpt='/home/cqjtu/GLIGEN/OUTPUT/test/tag00/checkpoint_00415001.pth'
ckpt = '/home/cqjtu/Documents/gligen_checkpoints/810/checkpoint_00495001.pth' # sample_1
# ckpt= '/media/cqjtu/PortableSSD/dataSet/GLIGEN/2023810/test/tag16/checkpoint_00355001.pth'  # camback
# ckpt = '/home/cqjtu/Documents/gligen_checkpoints/810/checkpoint_00255001.pth'  # CAM_BACK_LEFT well
# ckpt = '/home/cqjtu/Documents/gligen_checkpoints/810/checkpoint_00460001.pth'  # CAM_BACK_RIGHT well
# ckpt = '/home/cqjtu/Documents/gligen_checkpoints/810/checkpoint_00490001.pth'  # CAM_FRONT bad
model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt)
prompt = 'a parking lot filled with lots of parked different red cars, red cars'
negative_prompt = ''
# longbody, lowres, bad anatomy, extra digit, ' \
#                 'fewer digits, cropped, worst quality, low quality

sampler = 1
ddim_steps = 35
num_samples = 2
scale = 5
# seed = 623418984
seed = 4648312541
eta = 0
strength = 0.9
alpha = 0.7


root = '/media/cqjtu/PortableSSD/dataSet/nusenes/nuscenes'
save_root='/home/cqjtu/Documents/dataset/gligen'
cams = [
    # 'CAM_BACK',
    # 'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
    'CAM_FRONT',
    'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
]
test_show = False
use_gradio = False
to_generation = True
sample_name = 'samples_1'
for cam in cams:
    if not os.path.exists(os.path.join(save_root, sample_name)):
        os.mkdir(os.path.join(save_root, sample_name))

    if not os.path.exists(os.path.join(save_root, sample_name, cam)):
        os.mkdir(os.path.join(save_root, sample_name, cam))
    image_rootdir = root + f'/sample_src/{cam}/img'
    depth_rootdir = root + f'/sample_src/{cam}/dep'
    sem_rootdir = root + f'/sample_src/{cam}/seg'
    caption_path =f'/media/cqjtu/PortableSSD/dataSet/nusenes/nuscenes/sample_src/{cam}/{sample_name}_new_caption.json'
    save_path =save_root + f'/{sample_name}/{cam}'
    print(f'load json from {caption_path}')
    gallery = inference(prompt, negative_prompt,
                        sampler, ddim_steps, num_samples, scale,
                        seed, eta, strength, alpha)

# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Stable Diffusion GLIGEN2Img")
#
#     with gr.Row():
#         with gr.Column():
#                 use_gradio=True
#             # dep = gr.Image(label='dep',source='upload', type="filepath")
#             # sem = gr.Image(label='sem',source='upload', type="filepath")
#             prompt = gr.Textbox(label="Prompt")
#             negative_prompt=gr.Textbox(label="negative_prompt")
#             run_button = gr.Button(label="Run")
#             with gr.Accordion("Advanced options", open=False):
#                 num_samples = gr.Slider(
#                     label="Images", minimum=1, maximum=4, value=1, step=1)
#                 sampler = gr.Slider(label="sampler 0:DDIM 1:PLMS", minimum=0,  maximum=1, value=0, step=1)
#                 ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=150, step=10)
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
#                 eta = gr.Number(label="eta (DDIM)", value=0.0,step=0.1)
#         with gr.Column():
#             # image = gr.Image(label='image',source='upload', type="pil")
#             gallery = gr.Gallery(label="Generated images", show_label=False).style(
#                 grid=[num_samples], height="auto",margin="auto")
#
#     run_button.click(fn=inference, inputs=[
#                      prompt, negative_prompt,
#                     sampler, ddim_steps, num_samples, scale,
#                     seed, eta, strength,alpha], outputs=[gallery])
#
#
# block.launch(share=True)
