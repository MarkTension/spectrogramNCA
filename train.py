"""
This code was adapted from Google Research's Colab notebook 
https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb
and modularized
"""

from NCA import CA
import imageio
from utils import imread, imshow, VideoWriter, grab_plot, zoom
import torch.nn.functional as F
import torchvision.models as models
import torch
import os
import numpy as np
import matplotlib.pylab as pl
from tqdm import tqdm
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# import vgg model
vgg16 = models.vgg16(weights='IMAGENET1K_V1').features.float()


def calc_styles_vgg(imgs):
    style_layers = [1, 6, 11, 18, 25]
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


def create_vgg_loss(target_img):
    yy = calc_styles_vgg(target_img)

    def loss_f(imgs):
        xx = calc_styles_vgg(imgs)
        return sum(ot_loss(x, y) for x, y in zip(xx, yy))
    return loss_f


def to_nchw(img):
    img = torch.as_tensor(img, dtype=torch.double)
    if len(img.shape) == 3:
        img = img[None, ...]
    return img.permute(0, 3, 1, 2)


def to_rgb(x):
    return x[..., :3, :, :]+0.5


def train(image_path: str):

    param_n = sum(p.numel() for p in CA().parameters())
    print('CA param count:', param_n)

    # target image
    # url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg'
    # style_img = imread(url, max_size=128)
    style_img = np.array(imageio.imread(image_path))
    style_img = style_img/255
    # w, h = style_img.shape 
    style_img = style_img[:,:style_img.shape[0], :3]


    with torch.no_grad():
        style_img = to_nchw(style_img).float()
        loss_f = create_vgg_loss(style_img)

    viz_img = style_img.cpu().numpy()
    imshow(np.moveaxis(viz_img[0,:,:], 0,-1), count=0)

    # setup training
    ca = CA()
    opt = torch.optim.Adam(ca.parameters(), 1e-3, capturable=False)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 2000], 0.3)
    loss_log = []
    with torch.no_grad():
        pool = ca.seed(256)

    # training loop
    gradient_checkpoints = False  # Set in case of OOM problems

    for i in range(5000):
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), 4, replace=False)
            x = pool[batch_idx]
            if i % 8 == 0:
                x[:1] = ca.seed(1)
        step_n = np.random.randint(32, 96)
        if not gradient_checkpoints:
            for k in range(step_n):
                x = ca(x)
        else:
            x.requires_grad = True
            x = torch.utils.checkpoint.checkpoint_sequential(
                [ca]*step_n, 16, x)

        overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
        loss = loss_f(to_rgb(x))+overflow_loss
        with torch.no_grad():
            loss.backward()
            for p in ca.parameters():
                p.grad /= (p.grad.norm()+1e-8)   # normalize gradients

            opt.step()
            opt.zero_grad()
            lr_sched.step()
            pool[batch_idx] = x                # update pool

            loss_log.append(loss.item())
            if i % 5 == 0:
                print(f" \
          step_n: {len(loss_log)} \
          loss: {loss.item()} \
          lr: {lr_sched.get_last_lr()[0]}")

            if i % 50 == 0:
                pl.plot(loss_log, '.', alpha=0.1)
                pl.yscale('log')
                pl.ylim(np.min(loss_log), loss_log[0])
                pl.tight_layout()
                imshow(grab_plot(), id='log', count=i)
                imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
                imshow(np.hstack(imgs), id='batch', count=i)

    print('done training')
    write_video(ca=ca)


def write_video(ca: CA):
    with VideoWriter() as vid, torch.no_grad():
        x = ca.seed(1, 256)
        for k in tqdm(range(300), leave=False):
            step_n = min(2**(k//30), 8)
            for _ in range(step_n):
                x[:] = ca(x)
                img = to_rgb(x[0]).permute(1, 2, 0).cpu().detach().numpy()

            vid.add(zoom(img, 2))
            del img

    print('done video')