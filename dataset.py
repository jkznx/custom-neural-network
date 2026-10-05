"""YOLO polygon loader, strict validation, native mask union and caching."""
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not images:
        raise ValueError(f'No images in {directory}')
    if len({p.stem for p in images}) != len(images):
        raise ValueError('Duplicate image stems would overwrite outputs')
    return images


def load_polygons(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing label {path}; negative images require an empty label file')
    polygons = []
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields)-1) % 2:
            raise ValueError(f'{path}:{n}: expected class + >=3 polygon points; boxes unsupported')
        values = np.asarray([float(x) for x in fields])
        if not np.isfinite(values).all() or values[0] != 0:
            raise ValueError(f'{path}:{n}: expected finite lane-only class 0')
        coords = values[1:].reshape(-1, 2)
        if ((coords < 0) | (coords > 1)).any():
            raise ValueError(f'{path}:{n}: coordinates outside [0,1]')
        area = abs(np.dot(coords[:, 0], np.roll(coords[:, 1], 1)) - np.dot(coords[:, 1], np.roll(coords[:, 0], 1))) / 2
        if area <= 1e-12:
            raise ValueError(f'{path}:{n}: degenerate polygon')
        polygons.append(coords)
    return polygons


def native_mask(path, size):
    w, h = size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in load_polygons(path):
        draw.polygon([(min(w-1, int(x*w)), min(h-1, int(y*h))) for x, y in polygon], fill=255)
    return mask


def image_tensor(image, height, width):
    image = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()/255).permute(2, 0, 1)


class LaneDataset(Dataset):
    def __init__(self, images_dir, labels_dir, height=96, width=160, augment=False):
        self.images = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.height, self.width, self.augment = height, width, augment
        self.cache = []
        for path in self.images:
            with Image.open(path) as source:
                image = source.convert('RGB')
            mask = native_mask(self.labels_dir/(path.stem+'.txt'), image.size)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            self.cache.append((np.asarray(image).copy(), (np.asarray(mask)>0).astype(np.uint8)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        array, mask = self.cache[index]
        image = Image.fromarray(array)
        if self.augment:
            a = array.astype(np.float32)
            if random.random() < .5:
                a *= np.random.uniform(.94, 1.06, (1, 1, 3))
            if random.random() < .5:
                a += np.random.uniform(-15, 15)
            image = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            if random.random() < .25:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .6)))
        return image_tensor(image, self.height, self.width), torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
