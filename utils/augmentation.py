from torchvision import datasets, transforms
from PIL import ImageFilter, ImageOps, Image, ImageDraw
import torchvision.transforms.functional as F
import random
import torch
from utils.pos_embed import get_2d_local_sincos_pos_embed

class RandomResizedCropCoord(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.ratio = ratio
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size), self.interpolation)
        return (i, j, h, w), img

    def __call__(self, img):
        return self.forward(img)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DataAugmentation(object):
    def __init__(self, scale=(0.2, 1.0), embed_dim=64, anchor_grid_size=56, query_grid_size=56):
        self.normalize = transforms.Compose([    
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.variance = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.4),
            Solarization(p=0.2),
        ])
        self.embed_dim = embed_dim
        self.anchor_grid_size = anchor_grid_size
        self.query_grid_size =query_grid_size
        
        self.rcr = RandomResizedCropCoord(224, scale=scale, interpolation=3)

    def calculate_sin_cos(self, lpos, gpos):
        kg = gpos[3] / self.anchor_grid_size
        w_bias = (lpos[1] - gpos[1]) / kg
        kl = lpos[3] / self.query_grid_size
        w_scale = kl / kg
        kg = gpos[2] / self.anchor_grid_size
        h_bias = (lpos[0] - gpos[0]) / kg
        kl = lpos[2] / self.query_grid_size
        h_scale = kl / kg
        return get_2d_local_sincos_pos_embed(self.embed_dim, self.query_grid_size, w_bias, w_scale, h_bias, h_scale).reshape(56, 56, -1)


    def forward(self, img):
        return self.__call__(img)

    def __call__(self, image):
        anchor_pos, anchor_img = self.rcr(image)
        target_pos, target_img = self.rcr(image)
        anchor_img = self.normalize(self.variance(anchor_img))
        target_img = self.normalize(target_img)
        target_pos_embed = torch.FloatTensor(self.calculate_sin_cos(target_pos, anchor_pos)).permute(2, 0, 1)
        return anchor_img, target_img, target_pos_embed

# if __name__ == '__main__':
#     aug = DataAugmentation()
#     img = Image.open("../figures/test.jpeg").convert('RGB')
#     outs = aug.forward(img)
#     print(outs[2].size())
