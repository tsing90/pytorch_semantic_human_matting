import cv2, os
import numpy as np
import argparse
import tqdm

"""
def get_args():
    parser = argparse.ArgumentParser(description='Trimap')
    parser.add_argument('--mskDir', type=str, required=True, help="masks directory")
    parser.add_argument('--saveDir', type=str, required=True, help="where trimap result save to")
    parser.add_argument('--list', type=str, required=True, help="list of images id")
    parser.add_argument('--size', type=int, required=True, help="kernel size")
    args = parser.parse_args()
    print(args)
    return args
"""
# a simple trimap generation code for fixed kernel size
def erode_dilate(msk, size=(10, 10), smooth=True):
    if smooth:
        size = (size[0]-4, size[1]-4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    dilated = cv2.dilate(msk, kernel, iterations=1)
    if smooth:  # if it is .jpg, prevent output to be jagged
        dilated[(dilated>5)] = 255
        dilated[(dilated <= 5)] = 0
    else:
        dilated[(dilated>0)] = 255

    eroded = cv2.erode(msk, kernel, iterations=1)
    if smooth:
        eroded[(eroded<250)] = 0
        eroded[(eroded >= 250)] = 255
    else:
        eroded[(eroded < 255)] = 0

    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128

    """# make sure there are only 3 values in trimap
    cnt0 = len(np.where(res >= 0)[0])
    cnt1 = len(np.where(res == 0)[0])
    cnt2 = len(np.where(res == 128)[0])
    cnt3 = len(np.where(res == 255)[0])
    assert cnt0 == cnt1 + cnt2 + cnt3
    """

    return res

# trimap generation with different/random kernel size
def rand_trimap(msk, smooth=False):
    h, w = msk.shape
    scale_up, scale_down = 0.022, 0.006   # hyper parameter
    dmin = 0        # hyper parameter
    emax = 255 - dmin   # hyper parameter

    #  .jpg (or low quality .png) tend to be jagged, smoothing tricks need to be applied
    if smooth:
        # give thrshold for dilation and erode results
        scale_up, scale_down = 0.02, 0.006
        dmin = 5
        emax = 255 - dmin

        # apply gussian smooth
        if h<1000:
            gau_ker = round(h*0.01)  # we restrict the kernel size to 5-9
            gau_ker = gau_ker if gau_ker % 2 ==1 else gau_ker-1  # make sure it's odd
            if h<500:
                gau_ker = max(3, gau_ker)
            msk = cv2.GaussianBlur(msk, (gau_ker, gau_ker), 0)

    kernel_size_high = max(10, round((h + w) / 2 * scale_up))
    kernel_size_low  = max(1, round((h + w) /2 * scale_down))
    erode_kernel_size  = np.random.randint(kernel_size_low, kernel_size_high)
    dilate_kernel_size = np.random.randint(kernel_size_low, kernel_size_high)

    erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    eroded_alpha = cv2.erode(msk, erode_kernel)
    dilated_alpha = cv2.dilate(msk, dilate_kernel)

    dilated_alpha = np.where(dilated_alpha > dmin, 255, 0)
    eroded_alpha = np.where(eroded_alpha < emax, 0, 255)

    res = dilated_alpha.copy()
    res[((dilated_alpha == 255) & (eroded_alpha == 0))] = 128

    return res


def get_trimap(msk, smooth=True):
    h, w = msk.shape[:2]
    scale_up, scale_down = 0.022, 0.008   # hyper parameter
    dmin = 0        # hyper parameter
    emax = 255 - dmin   # hyper parameter

    #  .jpg (or low quality .png) tend to be jagged, smoothing tricks need to be applied
    if smooth:
        scale_up, scale_down = 0.02, 0.006
        dmin = 5
        emax = 255 - dmin

    kernel_size_high = max(10, round(h * scale_up))
    kernel_size_low  = max(1, round(h * scale_down))
    kernel_size = (kernel_size_high + kernel_size_low)//2

    print('kernel size:', kernel_size)

    erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_alpha = cv2.erode(msk, erode_kernel)
    dilated_alpha = cv2.dilate(msk, dilate_kernel)

    dilated_alpha = np.where(dilated_alpha > dmin, 255, 0)
    eroded_alpha = np.where(eroded_alpha < emax, 0, 255)

    res = dilated_alpha.copy()
    res[((dilated_alpha == 255) & (eroded_alpha == 0))] = 128

    return res


def main():
    kernel_size = 18
    alphaDir = 'alpha'  # where alpha matting images are
    trimapDir = 'trimap'
    names = os.listdir('image')
    if names == []:
        raise ValueError('No images are in the dir: ./image')
    print("Images Count: {}".format(len(names)))

    for name in tqdm.tqdm(names):
        alpha_path = alphaDir + "/" + name
        trimap_path = trimapDir + "/" + name.strip()[:-4] + ".png"  # output must be .png format
        alpha = cv2.imread(alpha_path, 0)
        if name[-3:] != 'png':
            trimap = erode_dilate(alpha, size=(kernel_size, kernel_size), smooth=True)
        else:
            trimap = erode_dilate(alpha, size=(kernel_size,kernel_size))

        #print("Write to {}".format(trimap_name))
        cv2.imwrite(trimap_path, trimap)

if __name__ == "__main__":
    main()


