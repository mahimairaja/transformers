# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for ViT."""

from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import os
import re
import cv2
import imageio

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...utils import (
    TensorType,
    is_scipy_available,
    is_torchvision_available,
    is_torch_available,
    is_vision_available,
    logging,
)


if is_torch_available():
    import torch


if is_torchvision_available():
    import torchvision
    from torchvision import transforms


if is_vision_available():
    from PIL import Image


if is_scipy_available():
    import scipy

logger = logging.get_logger(__name__)


class ProPainterImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ProPainter image processor.

    Args:
        TODO
    """

    def __init__(
        self,
        video: str,
        mask: str,
        mode: str,
        scale_h: float = 1.0,
        scale_w: float = 1.2,
        resize_ratio: float = 1.0,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = {"height": -1, "width": -1},
        mask_dilation: int = 4,
        save_frames: bool = False,
        save_path: str = 'results',
        save_fps: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.video = video
        self.mask = mask
        self.mode = mode
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.resize_ratio = resize_ratio
        self.do_resize = do_resize
        self.size = size
        self.mask_dilation = mask_dilation
        self.save_frames = save_frames
        self.save_path = save_path
        self.save_fps = save_fps


    def to_tensors(self) -> torch.Tensor:
        return transforms.Compose([Stack(), ToTorchFormatTensor()])

    def get_device(
        self,
        gpu_id : str = None
    ) -> torch.device:
        """
        TODO
        """
        IS_HIGH_VERSION = [int(m) for m in list(re.findall(r"^([0-9]+)\.([0-9]+)\.([0-9]+)([^0-9][a-zA-Z0-9]*)?(\+git.*)?$",\
            torch.__version__)[0][:3])] >= [1, 12, 0]

        if gpu_id is None:
            gpu_str = ''
        elif isinstance(gpu_id, int):
            gpu_str = f':{gpu_id}'
        else:
            raise TypeError('Input should be int value.')

        if IS_HIGH_VERSION:
            if torch.backends.mps.is_available():
                return torch.device('mps'+gpu_str)
        return torch.device('cuda'+gpu_str if torch.cuda.is_available() and torch.backends.cudnn.is_available() else 'cpu')

    def resize_frames(
        self,
        frames,
        size=None
    ) -> Tuple[List[Image.Image], Tuple[int, int], Tuple[int, int]]:
        """
        TODO
        """
        if size is not None:
            out_size = size
            process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
            frames = [f.resize(process_size) for f in frames]
        else:
            out_size = frames[0].size
            process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
            if not out_size == process_size:
                frames = [f.resize(process_size) for f in frames]

        return frames, process_size, out_size


    def read_frame_from_videos(
        self,
        frame_root : str
    ) -> Tuple[List[Image.Image], Optional[float], Tuple[int, int], str]:
        """
        TODO
        """
        if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
            video_name = os.path.basename(frame_root)[:-4]
            vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
            frames = list(vframes.numpy())
            frames = [Image.fromarray(f) for f in frames]
            fps = info['video_fps']
        else:
            video_name = os.path.basename(frame_root)
            frames = []
            fr_lst = sorted(os.listdir(frame_root))
            for fr in fr_lst:
                frame = cv2.imread(os.path.join(frame_root, fr))
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame)
            fps = None
        size = frames[0].size

        return frames, fps, size, video_name

    def binary_mask(
        self,
        mask: np.ndarray,
        th: float = 0.1
    ) -> np.ndarray:
        """
        TODO
        """
        mask[mask>th] = 1
        mask[mask<=th] = 0
        return mask

    def read_mask(
        self,
        mpath,
        length,
        size,
        flow_mask_dilates : int = 8,
        mask_dilates : int = 5
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        TODO
        """
        masks_img = []
        masks_dilated = []
        flow_masks = []

        if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            masks_img = [Image.open(mpath)]
        else:
            mnames = sorted(os.listdir(mpath))
            for mp in mnames:
                masks_img.append(Image.open(os.path.join(mpath, mp)))

        for mask_img in masks_img:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))

            # Dilate 8 pixel so that all known pixel is trustworthy
            if flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
            else:
                flow_mask_img = self.binary_mask(mask_img).astype(np.uint8)
            # Close the small holes inside the foreground objects
            flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
            flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))

            if mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = self.binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))

        if len(masks_img) == 1:
            flow_masks = flow_masks * length
            masks_dilated = masks_dilated * length

        return flow_masks, masks_dilated

    def extrapolation(
        self,
        video_ori,
        scale : Tuple
    ) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], Tuple[int, int]]:
        """Prepares the data for video outpainting.
        """
        nFrame = len(video_ori)
        imgW, imgH = video_ori[0].size

        # Defines new FOV.
        imgH_extr = int(scale[0] * imgH)
        imgW_extr = int(scale[1] * imgW)
        imgH_extr = imgH_extr - imgH_extr % 8
        imgW_extr = imgW_extr - imgW_extr % 8
        H_start = int((imgH_extr - imgH) / 2)
        W_start = int((imgW_extr - imgW) / 2)

        # Extrapolates the FOV for video.
        frames = []
        for v in video_ori:
            frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
            frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
            frames.append(Image.fromarray(frame))

        # Generates the mask for missing region.
        masks_dilated = []
        flow_masks = []

        dilate_h = 4 if H_start > 10 else 0
        dilate_w = 4 if W_start > 10 else 0
        mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

        mask[H_start+dilate_h: H_start+imgH-dilate_h,
            W_start+dilate_w: W_start+imgW-dilate_w] = 0
        flow_masks.append(Image.fromarray(mask * 255))

        mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
        masks_dilated.append(Image.fromarray(mask * 255))

        flow_masks = flow_masks * nFrame
        masks_dilated = masks_dilated * nFrame

        return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)

    def preprocess(
        self,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            TODO

        """

        device = self.get_device()

        frames, fps, size, video_name = self.read_frame_from_videos(self.video)
        if not self.size['width'] == -1 and not self.size['height'] == -1:
            size = (self.size['width'], self.size['height'])
        if not self.resize_ratio == 1.0:
            size = (int(self.resize_ratio * size[0]), int(self.resize_ratio * size[1]))

        if self.do_resize:
            frames, size, out_size = self.resize_frames(frames, size)

        if self.mode == 'video_inpainting':
            frames_len = len(frames)
            flow_masks, masks_dilated = self.read_mask(self.mask, frames_len, size,
                                                flow_mask_dilates=self.mask_dilation,
                                                mask_dilates=self.mask_dilation)
        elif self.mode == 'video_outpainting':
            frames, flow_masks, masks_dilated, size = self.extrapolation(frames, (self.scale_h, self.scale_w))
        else:
            raise NotImplementedError

        frames = self.to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = self.to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = self.to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

        data = {"frames": frames, "flow_masks": flow_masks, "distil_masks": masks_dilated}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def save_videos_frame(
        self,
        save_root : str,
        masked_frame_for_save,
        comp_frames,
    ) -> None:
        """
        Save the frames of the video.

        Args:
            TODO

        """
        imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=self.save_fps, quality=7)
        imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=self.save_fps, quality=7)
        return None

    def imwrite(
        self,
        img,
        file_path,
        params=None,
        auto_mkdir=True
    ) -> bool:
        if auto_mkdir:
            dir_name = os.path.abspath(os.path.dirname(file_path))
            os.makedirs(dir_name, exist_ok=True)

        return cv2.imwrite(file_path, img, params)


    def save_frame(
        self,
        comp_frames,
        video_length,
        out_size,
        save_root : str,
    ) -> None:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            self.imwrite(f, img_save_root)

        return None


    def post_process(
        self,
        video_name: str,
        comp_frames,
        video_length,
        out_size,
    ) -> None:
        """
        Postporcess the outputs of the model.

        Args:
            TODO

        """
        # Save the frames of the video.
        save_root = os.path.join(self.save_path, video_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        if self.save_frames:
            self.save_frame(comp_frames, video_length, out_size, save_root)

        # Save the video.
        if self.mode == 'video_outpainting':
            comp_frames = [i[10:-10,10:-10] for i in comp_frames]
            masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
        elif self.mode == 'video_inpainting':
            masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
            comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        else:
            raise NotImplementedError(f"Image mode {self.mode}")

        self.save_videos_frame(save_root, masked_frame_for_save, comp_frames)

        return None

class Stack(object):
    def __init__(self, roll=False) -> None:
        self.roll = roll

    def __call__(self, img_group) -> np.ndarray:
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True) -> None:
        self.div = div

    def __call__(self, pic) -> torch.Tensor:
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img
