# UTF-8 encoded
import lpips
import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from itertools import accumulate
from pathlib import Path
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import Dataset, Subset
from typing import Sequence, Union

lpips_model = None


# def angle2mat(angle):
#     angle = angle.reshape(-1)
#     sinx, siny, sinz = np.sin(angle)
#     cosx, cosy, cosz = np.cos(angle)
#     rotx = np.eye(3)
#     rotx[1, 1], rotx[1, 2], rotx[2, 1], rotx[2, 2] = cosx, -sinx, sinx, cosx
#     roty = np.eye(3)
#     roty[0, 0], roty[0, 2], roty[2, 0], roty[2, 2] = cosy, siny, -siny, cosy
#     rotz = np.eye(3)
#     rotz[0, 0], rotz[0, 1], rotz[1, 0], rotz[1, 1] = cosz, -sinz, sinz, cosz
#     return rotz @ roty @ rotx


# def to8b(img):
#     if isinstance(img, torch.Tensor):
#         img = img.detach().cpu().numpy()
#     elif not isinstance(img, np.ndarray):
#         img = np.asarray(img)
#     if img.shape[0] in [1, 3]:
#         img = img.transpose((1, 2, 0)).squeeze()
#     return (np.clip(img, 0., 1.) * 255).astype(np.uint8)


def str2bool(s: str):
    return s.lower() in ("true", "t", "1", "yes", "y")


def render_to_rgb(figure, close=True):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    if close:
        plt.close(figure)
    return image_hwc


def calcSSIM(x, y, data_range=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().clip(0, 1.)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    assert x.ndim == y.ndim == 3
    channel_axis = 0 if x.shape[0] in [1, 3] else 2
    if data_range is None:
        data_range = 1.
    return structural_similarity(x, y, data_range=data_range, channel_axis=channel_axis)


def calcPSNR(x, y, data_range=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().clip(0, 1.)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    assert x.ndim == y.ndim == 3
    if data_range is None:
        data_range = 1.
    return peak_signal_noise_ratio(x, y, data_range=data_range)


@torch.no_grad()
def calcLPIPS(x, y):
    if x.ndim == 3:
        x = x[None].clamp(0, 1.)
    if y.ndim == 3:
        y = y[None]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.ndim == 4 and y.ndim == 4
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS()
    lpips_model = lpips_model.to(x.device)
    return lpips_model(x, y, normalize=True).item()


# class VGGLoss(torch.nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()
#         self.vgg_net = lpips.LPIPS(net='vgg')
#
#     def forward(self, x, y, normalize=True):
#         # default input is [0,1], if not, set normalize to False
#         val = self.vgg_net(x, y, normalize=normalize)
#         return val.mean()
#
#
# class LPIPSLoss(torch.nn.Module):
#     def __init__(self):
#         super(LPIPSLoss, self).__init__()
#         self.lpips_model = lpips.LPIPS(net="alex")
#
#     def forward(self, x, y, normalize=True):
#         # default input is [0,1], if not, set normalize to False
#         val = self.lpips_model(x, y, normalize=normalize)
#         return val.mean()
