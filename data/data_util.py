# code for temporary usage or previously used
# author: L.Q. Chen

import cv2
import os
import random
import numpy as np
import math

# for convenience, we make fake fg and bg image
def fake_fg_bg(img,alpha):
    color_fg = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    color_bg = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    fg = np.full(img.shape, color_fg)
    bg = np.full(img.shape, color_bg)
    a = np.expand_dims(alpha/255., axis=2)

    fg = img * a + fg * (1-a)
    bg = bg * a + img * (1-a)
    return fg.astype(np.uint8), bg.astype(np.uint8)

# composite fg and bg with arbitrary size
def composite(image, a, bg):
    h, w = a.shape
    bh, bw = bg.shape[:2]
    wratio, hratio = w/bw, h/bh
    ratio = wratio if wratio > hratio else hratio
    bg = cv2.resize(bg, (math.ceil(bw * ratio), math.ceil(bh * ratio)), cv2.INTER_CUBIC)
    bh, bw = bg.shape[:2]
    assert bh>=h and bw>=w
    bg = np.array(bg[(bh - h) // 2: (bh - h) // 2 + h, (bw - w) // 2: (bw - w) // 2 + w], np.float32)
    assert h, w == bg.shape[:2]

    fg = np.array(image, np.float32)
    alpha = np.expand_dims(a/255., axis=2).astype(np.float32)
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)

    # we need bg for calculating compositional loss in M_net
    return comp, bg

def read_files(name, return_a_bg=True):
    if name[2] == 'bg':
        img_path, bg_path = name[0].strip(), name[1].strip()
        assert os.path.isfile(img_path) and os.path.isfile(bg_path), (img_path, bg_path)

        image = cv2.imread(img_path, -1)  # it's RGBA image
        fg = image[:,:,:3]
        a = image[:,:,3]
        bg = cv2.imread(bg_path)

        image, bg = composite(fg, a, bg)
        trimap = gen_trimap.rand_trimap(a)

    elif name[2] == 'msk':
        img_path, a_path = name[0].strip(), name[1].strip()
        assert os.path.isfile(img_path) and os.path.isfile(a_path)

        image = cv2.imread(img_path)  # it's composited image
        a = cv2.imread(a_path, 0)  # it's grayscale image

        a[a>0] = 255
        trimap = gen_trimap.rand_trimap(a)

        # for M-net and fusion training, we need bg for compositional loss
        # here to simplify, we convert original image
        if return_a_bg:
            fg, bg = fake_fg_bg(image, a)

    # NOTE ! ! ! trimap should be 3 classes for classification : fg, bg. unsure
    trimap[trimap == 0] = 0
    trimap[trimap == 128] = 1
    trimap[trimap == 255] = 2

    assert image.shape[:2] == trimap.shape[:2] == a.shape[:2]

    if return_a_bg:
        return image, trimap, a, bg, fg
    else:
        return image, trimap

# crop image/alpha/trimap with random size of [max_size//2, max_size] then resize to patch_size
def random_patch(image, trimap, patch_size, alpha=None, bg=None, fg=None):
    h, w = image.shape[:2]
    max_size = max(h, w)
    min_size = max_size//2
    if isinstance(alpha, np.ndarray) and isinstance(bg, np.ndarray):
        patch_a_bg= True
    else:
        patch_a_bg = False

    count = 0 # debug usage
    while True:
        sqr_tri = np.zeros((max_size, max_size), np.uint8)
        sqr_img = np.zeros((max_size, max_size, 3), np.uint8)
        if patch_a_bg:
            sqr_alp = np.zeros((max_size, max_size), np.uint8)
            sqr_bg = np.zeros((max_size, max_size, 3), np.uint8)
            sqr_fg = np.zeros((max_size, max_size, 3), np.uint8)
        if h>=w:
            sqr_tri[:, (h-w)//2 : (h-w)//2+w] = trimap
            sqr_img[:, (h-w)//2 : (h-w)//2+w] = image
            if patch_a_bg:
                sqr_alp[:, (h-w)//2 : (h-w)//2+w] = alpha
                sqr_bg[:, (h-w)//2 : (h-w)//2+w] = bg
                sqr_fg[:, (h-w)//2 : (h-w)//2+w] = fg
        else:
            sqr_tri[(w-h)//2 : (w-h)//2+h, :] = trimap
            sqr_img[(w-h)//2 : (w-h)//2+h, :] = image
            if patch_a_bg:
                sqr_alp[(w-h)//2 : (w-h)//2+h, :] = alpha
                sqr_bg[(w-h)//2 : (w-h)//2+h, :] = bg
                sqr_fg[(w-h)//2 : (w-h)//2+h, :] = fg

        crop_size = random.randint(min_size, max_size)  # both value are inclusive
        x = random.randint(0, max_size-crop_size)    # 0 is inclusive
        y = random.randint(0, max_size-crop_size)
        trimap_temp = sqr_tri[y: y+crop_size, x: x+crop_size]
        if len(np.where(trimap_temp == 1)[0])>0:  # check if unknown area is included
            image = sqr_img[y: y+crop_size, x: x+crop_size]
            trimap = trimap_temp
            if patch_a_bg:
                alpha = sqr_alp[y: y+crop_size, x: x+crop_size]
                bg = sqr_bg[y: y+crop_size, x: x+crop_size]
                fg = sqr_fg[y: y+crop_size, x: x+crop_size]
            break
        elif len(np.where(trimap==1)[0]) == 0:
            print('Warning & Error: No unknown area in current trimap! Refer to saved trimap in <exception> folder.')
            image = sqr_img
            trimap = sqr_tri
            if patch_a_bg:
                alpha = sqr_alp
                bg = sqr_bg
                fg = sqr_fg
            os.makedirs('ckpt/exceptions', exist_ok=True)
            cv2.imwrite('ckpt/exceptions/img_{}_{}.png'.format(str(h), str(w)), image)
            cv2.imwrite('ckpt/exceptions/tri_{}_{}.png'.format(str(h), str(w)), trimap)
            break
        elif count>5:
            print('Warning: cannot find right patch randomly, use max_size instead! Refer to saved files in <exceptions> folder.')
            image = sqr_img
            trimap = sqr_tri
            if patch_a_bg:
                alpha = sqr_alp
                bg = sqr_bg
                fg = sqr_fg
            os.makedirs('ckpt/exceptions', exist_ok=True)
            cv2.imwrite('ckpt/exceptions/img_{}_{}.png'.format(str(h), str(w)), image)
            cv2.imwrite('ckpt/exceptions/tri_{}_{}.png'.format(str(h), str(w)), trimap)
            break

        count += 1 # debug usage

    image = cv2.resize(image, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    if patch_a_bg:
        alpha = cv2.resize(alpha, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        bg = cv2.resize(bg, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        fg = cv2.resize(fg, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        return image, trimap, alpha, bg, fg
    else:
        return image, trimap