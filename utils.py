import numpy as np
import math
import torch
from PIL import Image

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def fill_image(x, block_size):
    c = x.size()[0]
    h = x.size()[1]
    h_lack = 0
    w = x.size()[2]
    w_lack = 0
    if h % block_size != 0:
        h_lack = block_size - h % block_size
        temp_h = torch.zeros(c, h_lack, w)
        h = h + h_lack
        x = torch.cat((x, temp_h), 1)

    if w % block_size != 0:
        w_lack = block_size - w % block_size
        temp_w = torch.zeros(c, h, w_lack)
        w = w + w_lack
        x = torch.cat((x, temp_w), 2)
    return x, h ,w

def read_img(path, mode = 'G'):
    if mode == 'G':
        image = Image.open(path).convert("L")
    else:
        image = Image.open(path).convert("RGB")
    Iorg = np.array(image, dtype='float32')  # 读图
    image = (Iorg/127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image)
    if mode == 'RGB':
        image = np.transpose(image,(2,0,1))
    else:
        image = torch.unsqueeze(image, 0)
    return image

def make_batch(image, block_size, device, channels = 3):
    x ,h ,w = fill_image(image, block_size)
    batchs = torch.unsqueeze(x, 0)
    batchs = batchs.to(memory_format=torch.contiguous_format).float()
    batchs = batchs.to(device)
    return batchs

def de_batch(batchs, block_size, h, w, channels = 3):
    num_patches = batchs.size()[0]
    x = torch.zeros(1, channels, h, w)
    idx_h = range(0, h, block_size)
    idx_w = range(0, w, block_size)
    count = 0
    for a in idx_h:
        for b in idx_w:
            x[:, :, a:a + block_size, b:b + block_size] = batchs[count, :, :, :]
            count = count + 1
    return x

def test_rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    rlt = rlt.round()

    return rlt.astype(in_img_type)

def read_img2(path, mode = 'G'):
    Iorg = np.array(Image.open(path), dtype='float32')
    # if mode == 'G':
    #     image = Image.open(path).convert("L")
    # else:
    #     image = Image.open(path).convert("RGB")
    if len(Iorg.shape) == 3: #rgb转y
        Iorg = test_rgb2ycbcr(Iorg)
    # Iorg = np.array(image, dtype='float32')  # 读图
    image = (Iorg/127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image)
    if mode == 'RGB':
        image = np.transpose(image,(2,0,1))
    else:
        image = torch.unsqueeze(image, 0)
    return image