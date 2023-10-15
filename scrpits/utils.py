import os

import PIL
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from ldm.util import instantiate_from_config
from torchvision.transforms import transforms
from trainer import batch_to_device
import random
import json
device = 'cuda:0'

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.
def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict(saved_ckpt['model'])
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return model, autoencoder, text_encoder, diffusion, config


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize((512, 512))
    return image


from dataset.dataset_sem_dep import recursively_read


class inference_data_load:
    def __init__(self, image_rootdir, depth_rootdir, sem_rootdir, caption_path, prob_use_caption=1, image_size=512,
                 random_flip=False):
        self.image_rootdir = image_rootdir
        self.depth_rootdir = depth_rootdir
        self.sem_rootdir = sem_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_flip = random_flip

        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['jpg'])
        image_files.sort()
        sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=['png'])
        sem_files.sort()
        dep_files = recursively_read(rootdir=depth_rootdir, must_contain="", exts=['png'])
        dep_files.sort()

        self.image_files = image_files
        self.sem_files = sem_files
        self.dep_file = dep_files
        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)

        assert len(self.image_files) == len(self.dep_file) == len(self.sem_files) == len(
            self.image_filename_to_caption_mapping)
        self.pil_to_tensor = transforms.PILToTensor()
    def __getitem__(self, index):

        image_path = self.image_files[index]

        out = {}

        out['id'] = index
        image = Image.open(image_path).convert("RGB")
        sem = Image.open(self.sem_files[index]).convert("L")  # semantic class index 0,1,2,3,4 in uint8 representation
        dep = Image.open(self.dep_file[index])
        assert image.size == sem.size

        sem = self.pil_to_tensor(sem)[0, :, :]
        depth = (self.pil_to_tensor(dep).float() / 255 - 0.5) / 0.5

        depth = depth.expand(3, -1, -1)  #
        input_label = torch.zeros(152, 900, 1600)
        sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

        out['image'] = (self.pil_to_tensor(image).float() / 255 - 0.5) / 0.5
        out['sem'] = sem
        out['depth'] = depth
        out['mask'] = torch.tensor(1.0)
        out['name']=image_path

        # -------------------- caption ------------------- #
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping[os.path.basename(image_path)]
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.image_files)

@torch.no_grad()
def prepare_batch_sem_depth(image, dep, sem, prompt, batch=1):
    pil_to_tensor = transforms.PILToTensor()

    init_image = Image.open(image).convert("RGB")
    init_image = init_image.resize((512, 512), resample=PIL.Image.LANCZOS)
    # init_image = TF.center_crop(init_image, min(init_image.size))
    # init_image = init_image.resize((512, 512), Image.NEAREST)
    init_image = (pil_to_tensor(init_image).float() / 255 - 0.5) / 0.5

    sem = Image.open(sem).convert("L")  # semantic class index 0,1,2,3,4 in uint8 representation
    # sem = TF.center_crop(sem, min(sem.size))
    # sem = sem.resize((512, 512), Image.NEAREST)  # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly

    sem = pil_to_tensor(sem)[0, :, :]
    input_label = torch.zeros(152, 900, 1600)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    depth = Image.open(dep).convert("RGB")
    # depth = crop_and_resize(depth)
    depth = (pil_to_tensor(depth).float() / 255 - 0.5) / 0.5

    out = {
        "caption": prompt,
        "image": init_image.unsqueeze(0).repeat(batch, 1, 1, 1),
        "sem": sem.unsqueeze(0).repeat(batch, 1, 1, 1),
        "depth": depth.unsqueeze(0).repeat(batch, 1, 1, 1),
        "mask": torch.ones(batch, 1),
    }
    return batch_to_device(out, device)


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=1 stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """
    if type == None:
        type = [1, 0, 0]

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale
