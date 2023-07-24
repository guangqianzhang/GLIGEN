import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from ldm.util import instantiate_from_config
from torchvision.transforms import transforms
from trainer import batch_to_device

device='cuda:0'
def load_ckpt(ckpt_path):

    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image


@torch.no_grad()
def prepare_batch_sem_depth(image,dep,sem, prompt,batch=1):
    pil_to_tensor = transforms.PILToTensor()

    init_image = image.convert("RGB")
    # init_image = TF.center_crop(init_image, min(init_image.size))
    # init_image = init_image.resize((512, 512), Image.NEAREST)
    init_image =(pil_to_tensor(init_image).float() / 255 - 0.5 ) / 0.5

    sem = Image.open(sem).convert("L")  # semantic class index 0,1,2,3,4 in uint8 representation
    # sem = TF.center_crop(sem, min(sem.size))
    # sem = sem.resize((512, 512), Image.NEAREST)  # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    # try:
    #     sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
    #     Image.fromarray(sem_color).save("sem_vis.png")
    # except:
    #     pass
    sem = pil_to_tensor(sem)[0, :, :]
    input_label = torch.zeros(152, 900, 1600)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    depth = Image.open(dep).convert("RGB")
    # depth = crop_and_resize(depth)
    depth = (pil_to_tensor(depth).float() / 255 - 0.5) / 0.5

    out = {
        "caption":prompt,
        "image":init_image.unsqueeze(0).repeat(batch,1,1,1),
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


