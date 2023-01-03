"""
This code is for generating textures with cellular automata. 
It is adapted from Google Research's Colab notebook
https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb
commented and modularized by me

The paper is neatly explained here: https://distill.pub/selforg/2021/textures/ 
"""

from NCA import CA
import imageio
from utils import imsave, VideoWriter, grab_plot, zoom, AttributeDict
import torch.nn.functional as F
import torchvision.models as models
import torch
import os
import numpy as np
import matplotlib.pylab as pl
from tqdm import tqdm
from stft_transformer import StftTransformer
from torch.nn import MSELoss

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# import vgg model
vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.float()


def plot_progress(loss_log: list, paths: AttributeDict, x: torch.tensor, i: int):
    """ plots training progress: intermediate results"""
    pl.plot(loss_log, '.', alpha=0.1)
    pl.yscale('log')
    pl.ylim(np.min(loss_log), loss_log[0])
    pl.tight_layout()
    # save loss plot
    imsave(grab_plot(), id='log', count=i, path=paths.nca_results)
    imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
    # save nca result
    imsave(np.hstack(imgs), id='batch',
           count=i, path=paths.nca_results)


def save_audio_progress(transformer:StftTransformer, complex_numbers:np.array, paths):
    recon_complex_numbers = transformer.inverse_convert_complex(complex_numbers)
    transformer.complex_coords = recon_complex_numbers
    transformer.complex_to_audio(paths.reconstructed_wav)

def calc_styles_vgg(imgs):
    style_layers = [1, 6, 11,18, 25] #
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (imgs-mean) / std
    b, c, h, w = x.shape
    features = [x.reshape(b, c, h*w)]
    for i, layer in enumerate(vgg16[:max(style_layers)+1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            features.append(x.reshape(b, c, h*w))
    return features


def project_sort(x, proj):
    return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]


def ot_loss(source, target, proj_n=32):
    ch, n = source.shape[-2:]
    projs = F.normalize(torch.randn(ch, proj_n), dim=0)
    source_proj = project_sort(source, projs)
    target_proj = project_sort(target, projs)
    target_interp = F.interpolate(target_proj, n, mode='nearest')
    return (source_proj-target_interp).square().sum()



# def create_mse_loss(target_img):
    
#     loss_f = MSELoss(target_img)

#     return loss_f


def create_vgg_loss(target_img):
    yy = calc_styles_vgg(target_img)

    def loss_f(imgs):
        xx = calc_styles_vgg(imgs)
        return sum(ot_loss(x, y) for x, y in zip(xx, yy))
    return loss_f
    
    
def create_image_loss(target_img):
    def loss_f(imgs):
        return torch.sum(torch.abs(target_img - imgs), keepdim=False, dim=None)
    return loss_f


def to_nchw(img):
    img = torch.as_tensor(img, dtype=torch.double)
    if len(img.shape) == 3:
        img = img[None, ...]
    return img.permute(0, 3, 1, 2)


def to_rgb(x):
    return x[..., :3, :, :]+0.5


def train(image, paths: AttributeDict, transformer:StftTransformer):
    """trains the neural cellular automata

    Args:
        image (np.array or path to image): the image to train on
        paths (AttributeDict): the paths for training
        transformer (StftTransformer): for transforming complex values to wav files
    """
    if (type(image) == str):
        style_img = np.array(imageio.imread(image))
        style_img = style_img/255
    elif (str(type(image)) == "<class 'numpy.ndarray'>"):
        style_img = image
    else:
        raise Exception("input error", "style image is not a path nor numpy array")

    # style_img = style_img[:500,:256,:]
    imsize = style_img.shape[:2]

    param_n = sum(p.numel() for p in CA().parameters())
    print('CA param count:', param_n)

    with torch.no_grad():
        style_img = to_nchw(style_img).float()
        loss_f = create_vgg_loss(style_img)
        loss_f_image = create_image_loss(style_img)

    viz_img = style_img.cpu().numpy()
    imsave(np.moveaxis(viz_img[0, :, :], 0, -1),
           count=0, path=paths.nca_results)

    # setup training
    ca = CA()
    opt = torch.optim.Adam(ca.parameters(), 1e-3, capturable=False)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 2000], 0.3)
    loss_log = []
    with torch.no_grad():
        pool = ca.seed(256, sz_w=imsize[0], sz_h=imsize[1]) # TODO: use the image size here instead of the default 128. 256 stands for the number of pools

    # training loop
    gradient_checkpoints = True  # Set in case of OOM problems

    for i in range(2000):
        loss, x, loss_log = train_step(pool, i, ca, gradient_checkpoints, loss_f_image, opt, lr_sched, loss_log, imsize)
        
        with torch.no_grad():
            if i % 5 == 0:
                print(f" \
            step_n: {len(loss_log)} \
            loss: {loss.item()} \
            lr: {lr_sched.get_last_lr()[0]}")

            if i % 50 == 0:
                plot_progress(loss_log, paths, x, i)

    print('done training')
    write_video(ca=ca, transformer=transformer, paths=paths, imsize=imsize)

def train_step(pool, i, ca, gradient_checkpoints, loss_f, opt, lr_sched, loss_log, imsize):
    """trains cellular automata for 1 step"""    
    with torch.no_grad():
        batch_idx = np.random.choice(len(pool), 4, replace=False) # sampling 4 out of 256. higher leads to OOM
        # TODO: get hardcoded values out of training step
        x = pool[batch_idx]
        if i % 8 == 0: # Prevent “catastrophic forgetting”: replace one sample in this batch with the original, single-pixel seed state
            # every 8 iterations we reset the first pool state. already randomly sampled
            x[:1] = ca.seed(1, sz_w=imsize[0], sz_h=imsize[1])

    step_n = np.random.randint(32, 96) # sample between 32 and 96 steps
    if not gradient_checkpoints:
        for _ in range(step_n):
            x = ca(x)
    else:
        x.requires_grad = True
        x = torch.utils.checkpoint.checkpoint_sequential(
            [ca]*step_n, 16, x)

    overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
    loss = loss_f(to_rgb(x)) + overflow_loss
    
    with torch.no_grad():
        loss.backward()
        for p in ca.parameters():
            p.grad /= (p.grad.norm()+1e-8)   # normalize gradients

        opt.step()
        opt.zero_grad()
        lr_sched.step()
        pool[batch_idx] = x                # update pool

        loss_log.append(loss.item())
    
    return loss, x, loss_log




def write_video(ca: CA, transformer:StftTransformer, paths:AttributeDict, imsize:tuple):
    with VideoWriter(filename=paths.nca_video) as vid, torch.no_grad():
        x = ca.seed(n=1, sz_w=imsize[0], sz_h=imsize[1])
        for k in tqdm(range(300), leave=False):
            step_n = min(2**(k//30), 8)
            for _ in range(step_n):
                x[:] = ca(x)
                img = to_rgb(x[0]).permute(1, 2, 0).cpu().detach().numpy()

            # vid.add(zoom(img, 2))
            vid.add(img)

            recon_complex_numbers = transformer.inverse_convert_complex(img[:,:,:2])
            transformer.complex_coords = recon_complex_numbers
            outname = f"{paths.nca_audio[:-4]}_{k}_{paths.nca_audio[-4:]}"
            transformer.complex_to_audio(outname)

            del img

    print('done video')
