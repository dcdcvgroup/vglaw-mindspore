import os
import mindspore as ms
from mindspore import ops
from mindformers import AutoTokenizer
import random
import numpy as np
import mindspore.dataset.vision as V
from PIL import ImageFilter, Image
import mindspore.dataset.transforms as T


def get_image_size(img):
    if isinstance(img, Image.Image):
        return img.size
    else:
        shape = img.shape
        if shape[0] == 3:  # (C,H,W)
            return [shape[-1], shape[-2]]
        elif shape[2] == 3:  # (H,W,C)
            return [shape[1], shape[0]]

    '''if isinstance(img, ms.Tensor):
        return [img.shape[-1], img.shape[-2]]
    else:
        return img.size'''


class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image, bbox, text, raw_text, ref_mask):
        for t in self.transforms:
            image, bbox, text, raw_text, ref_mask = t(image, bbox, text, raw_text, ref_mask)
        return image, bbox, text, raw_text, ref_mask

    def __repr__(self):
        format_string = '{}({}\n)'.format(
            self.__class__.__name__,
            ''.join(f'\n\t{t}' for t in self.transforms)
        )
        return format_string


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, p=0.6):
        self.prob = p
        self.jitter = V.RandomColorAdjust(brightness=brightness,
                                          contrast=contrast,
                                          saturation=saturation)

    def __call__(self, image, bbox, text, raw_text, ref_mask):
        if np.random.rand(1) < self.prob:
            image = self.jitter(image)

        return image, bbox, text, raw_text, ref_mask


class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.0), p=0):
        self.prob = p
        self.sigma = sigma

    def __call__(self, image, bbox, text, raw_text, ref_mask):
        if np.random.rand(1) < self.prob:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image, bbox, text, raw_text, ref_mask


# 好像存在问题
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self, image, bbox, text, raw_text, ref_mask=None):
        if np.random.rand(1) < self.prob:
            text = str(text)

            if isinstance(image, np.ndarray):
                img = V.HorizontalFlip()(image)
            else:
                img = V.Decode()(np.array(image))
                img = V.HorizontalFlip()(img)
            text = text.replace('right', '*&^special^&*')\
                .replace('left', 'right')\
                .replace('*&^special^&*', 'left')

            image = img

            if ref_mask is not None:
                if isinstance(ref_mask, np.ndarray):
                    ref_mask = V.HorizontalFlip()(ref_mask)
                else:
                    ref_mask = V.Decode()(ref_mask)
                    ref_mask = V.HorizontalFlip()(ref_mask)

            if bbox is not None:
                bbox[..., 0] = get_image_size(img)[0] - bbox[..., 0]

        return image, bbox, np.array(text), raw_text, ref_mask


class RandomResize(object):
    def __init__(self, img_size, *sizes, range=False):
        assert len(sizes) > 0, 'Sizes should not be empty.'
        self.img_size = img_size
        self.range = range
        if not range:
            self.sizes = sizes
        else:
            self.sizes = (min(sizes), max(sizes))

    def __call__(self, image, bbox, text, raw_text, ref_mask=None):
        if self.range:
            size = random.uniform(self.sizes[0], self.sizes[1])
        else:
            size = random.choice(self.sizes)

        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        text = text
        if any([wd in text for wd in dir_words]) and size > self.img_size:
            size = self.img_size
        img_w, img_h = get_image_size(image)
        ratio = 1.0 * size / max(img_h, img_w)

        if (bbox[2:] * ratio > self.img_size).any():
            size = self.img_size
            ratio = 1.0 * size / max(img_h, img_w)

        img_h, img_w = round(img_h * ratio), round(img_w * ratio)
        image = V.Resize([img_h, img_w])(image) # PIL
        
        if ref_mask is not None:
            ref_mask = ref_mask.transpose(1, 2, 0) # HWC
            ref_mask = V.Resize([img_h, img_w], V.Inter.NEAREST)(ref_mask)

        if bbox is not None:
            bbox = bbox * ratio

        return image, bbox, text, raw_text, ref_mask


class ToTensor(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, image, bbox, text, raw_text, ref_mask):
        image = V.ToTensor()(image) # array
        return image, bbox, text, raw_text, ref_mask


class NormalizeAndPad(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=512, translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.translate = translate

    def __call__(self, image, bbox, text, raw_text, ref_mask=None):
        img = image
        img = img.transpose(0, 2, 3, 1)
        img = ms.Tensor(V.Normalize(mean=self.mean, std=self.std)(img.numpy()))
        img = img.permute(0, 3, 1, 2)

        img_w, img_h = get_image_size(img)
        max_y = self.size - img_h
        max_x = self.size - img_w

        if self.translate:
            sy = np.random.randint(0, max_y + 1, (1, )).item()
            sx = np.random.randint(0, max_x + 1, (1, )).item()
        else:
            sy = max_y // 2
            sx = max_x // 2

        pad_img = ms.numpy.zeros((1, 3, self.size, self.size))
        pad_img[..., sy:sy+img_h, sx:sx+img_w] = img
        image = pad_img

        mask = ms.numpy.zeros((self.size, self.size))
        mask[sy:sy + img_h, sx:sx + img_w] = 1.0
        mask = mask.unsqueeze(0)

        if ref_mask is not None:
            ref_mask = ref_mask
            pad_ref_mask = ms.numpy.zeros((1, self.size, self.size))
            pad_ref_mask[..., sy:sy+img_h, sx:sx+img_w] = ref_mask
            ref_mask = pad_ref_mask

        if bbox is not None:
            bbox = bbox + ms.Tensor([sx, sy, 0, 0])
            bbox = bbox / ms.Tensor([self.size] * 4)
            bbox = ms.Tensor(bbox.unsqueeze(0), dtype=ms.float32)

        return image, bbox, text, raw_text, ref_mask, mask

class NormalizeAndPad_AllNumpy(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=512, translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.translate = translate

    def __call__(self, image, bbox, text, raw_text, ref_mask=None):
        img = image # (c,h,w)
        img_shape = img.shape
        if img_shape[0] == 3:
            is_hwc = False
        img = V.Normalize(mean=self.mean, std=self.std, is_hwc=is_hwc)(img)

        img_w, img_h = get_image_size(img)
        max_y = self.size - img_h
        max_x = self.size - img_w

        if self.translate:
            sy = np.random.randint(0, max_y + 1, (1, )).item()
            sx = np.random.randint(0, max_x + 1, (1, )).item()
        else:
            sy = max_y // 2
            sx = max_x // 2

        pad_img = np.zeros((3, self.size, self.size))
        pad_img[..., sy:sy+img_h, sx:sx+img_w] = img

        mask = np.zeros((self.size, self.size))
        mask[sy:sy + img_h, sx:sx + img_w] = 1.0

        if ref_mask is not None:
            pad_ref_mask = np.zeros((1, self.size, self.size))
            pad_ref_mask[..., sy:sy+img_h, sx:sx+img_w] = ref_mask.transpose(2,0,1)
            ref_mask = pad_ref_mask

        if bbox is not None:
            bbox = bbox + np.array([sx, sy, 0, 0])
            bbox = bbox / np.array([self.size] * 4)

        return pad_img, bbox, text, raw_text, ref_mask, mask


class RandomCrop(object):
    def __init__(self, img_size, min_size: int, max_size: int, crop_bbox=False):
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.crop_bbox = crop_bbox

    def __call__(self, image, bbox, text, raw_text, ref_mask=None):
        dir_words = ['left', 'right', 'top', 'bottom', 'middle']
        if any([wd in text for wd in dir_words]):
            return image, bbox, text, raw_text, ref_mask

        img = image

        img_w, img_h = get_image_size(img)

        cx, cy, w, h = list(bbox)
        if self.crop_bbox:
            ltx, lty, brx, bry = map(round, (cx, cy, cx, cy))
        else:
            ltx, lty, brx, bry = map(round, (cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h))

        tw = np.random.randint(min(img_w, self.min_size), min(img_w, self.max_size)+1)
        th = np.random.randint(min(img_h, self.min_size), min(img_h, self.max_size)+1)

        max_x, max_y = min(ltx, img_w - tw), min(lty, img_h - th)
        min_x, min_y = max(0, brx - tw), max(0, bry - th)

        if max_x < min_x or max_y < min_y:
            if img_w > self.img_size or img_h > self.img_size:
                print('#'*5, (tw, th), (img_w, img_h), bbox, '<>'*5, (max_x, max_y), (min_x, min_y), '<>'*5, (ltx, lty, brx, bry))
            return image, bbox, text, raw_text, ref_mask

        sx = np.random.randint(min_x, max_x + 1)
        sy = np.random.randint(min_y, max_y + 1)

        img = V.Crop([sy, sx], [th, tw])(img)
        image = img

        if ref_mask is not None:
            ref_mask = V.Crop([sy, sx], [th, tw])(ref_mask)

        offset = np.array([sx, sy])
        lt, br = bbox[:2] - 0.5 * bbox[2:] - offset, bbox[:2] + 0.5 * bbox[2:] - offset
        lt = np.array([max(lt[0], 0), max(lt[1], 0)])
        br = np.array([min(br[0], tw), min(br[1], th)])
        bbox = np.concatenate((0.5 * (lt + br), (br - lt)))

        return image, bbox, text, raw_text, ref_mask


class MyTokenizer(object):
    """
    Tokenize the texts.

    Args:
        bert_model_root(str): The path of the root of the bert model.
        bert_model(str): The type of the bert model.
        batch_size(int): The batch size.
        token_max_len(int): Max length of the final token.
        text_encoder_layer_num

    """

    def __init__(self,
                 bert_model='bert_base_uncased',
                 token_max_len=40,):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_len = token_max_len

    def __call__(self, texts):
        texts = texts.tolist()
        texts = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True)  # texts is a dict
        texts = [v for k, v in texts.items()]  # convert texts to a list
        return np.array(texts)