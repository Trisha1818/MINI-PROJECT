import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import closed_form_matting

# Select shadow image
image = 'snow.png'

# Load shadow image
shadow_img = cv2.cvtColor(cv2.imread('Samples/ShadowImages/' + image), cv2.COLOR_BGR2RGB).astype('float64') / 255.0
shadow_img = cv2.resize(shadow_img, (shadow_img.shape[1] // 2, shadow_img.shape[0] // 2))
plt.figure(); plt.imshow(shadow_img); plt.pause(2)

# Load hard mask
hard_mask = cv2.cvtColor(cv2.imread('Samples/HardMasks/' + image), cv2.COLOR_BGR2RGB).astype('float64') / 255.0
hard_mask = cv2.resize(hard_mask, (hard_mask.shape[1] // 2, hard_mask.shape[0] // 2))
plt.figure(); plt.imshow(hard_mask); plt.pause(2)

# Load scribbles
scribbles = cv2.imread('Samples/Scribbles/' + image, cv2.IMREAD_COLOR).astype('float64') / 255.0
scribbles = cv2.resize(scribbles, (scribbles.shape[1] // 2, scribbles.shape[0] // 2))

# Get soft mask using closed-form matting
soft_mask = closed_form_matting.closed_form_matting_with_scribbles(shadow_img, scribbles)
soft_mask = 1 - soft_mask
plt.figure(); plt.imshow(soft_mask); plt.pause(2)

# ---- Helper Functions ----

def computeAverage(arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        arr[arr == 0] = np.nan
        arr_mean = np.nanmean(arr, axis=(0, 1))
    return arr_mean

def getIntensityRatio(imgPatch, maskPatch, softMaskPatch):
    s = imgPatch
    m = maskPatch

    shd = s * m
    non_shd = s * (1 - m)

    shd_mean = computeAverage(shd)
    non_shd_mean = computeAverage(non_shd)

    shd_K = softMaskPatch * m
    non_shd_K = softMaskPatch * (1 - m)

    shd_mean_K = computeAverage(shd_K)
    non_shd_mean_K = computeAverage(non_shd_K)

    denominator = (shd_mean * non_shd_mean_K) - (non_shd_mean * shd_mean_K)
    denominator[denominator == 0] = 1e-5

    r = (non_shd_mean - shd_mean) / denominator
    r = np.nan_to_num(r)

    return r

def getMaskRatio(m):
    mask_gray = cv2.cvtColor((m * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    mOnes = np.count_nonzero(mask_gray > 128)
    totalCount = mask_gray.size
    mZeros = totalCount - mOnes
    patchRatio = mOnes / (mOnes + mZeros)
    return patchRatio

def getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height):
    xE = min(x + patch_size, height)
    yE = min(y + patch_size, width)

    soft_m = soft_mask[x:xE, y:yE]
    soft_m = np.repeat(soft_m[:, :, np.newaxis], 3, axis=2)

    m = hard_mask[x:xE, y:yE, :]
    s = shadow_img[x:xE, y:yE, :]

    maskRatio = getMaskRatio(m)

    if 0.49 < maskRatio < 0.51:
        return s, m, soft_m, True
    else:
        return None, None, None, False

def updateBins(bins, binIndx, r):
    for i in range(3):
        currentR = round(r[i], 1)
        bin_idx = int(currentR * 10)
        if 0 <= bin_idx < len(binIndx):
            bins[bin_idx, i] += 1
    return bins

def getFinalRatio(bins, binIndx):
    r = np.zeros(3)
    for i in range(3):
        max_idx = np.argmax(bins[:, i])
        r[i] = binIndx[max_idx]
    return r

def fixPatchShadow(imgPatch, maskPatch, softMaskPatch, ratio):
    mapper = (ratio + 1) / (ratio * softMaskPatch + 1e-5)
    fixed = imgPatch * mapper
    return fixed

def shadowRemover(shadow_img, soft_mask, hard_mask, patch_size=12, offset=10):
    height, width = shadow_img.shape[:2]

    binIndx = np.arange(0, 100, 0.1)
    bins = np.zeros((len(binIndx), 3))

    for x in range(0, height - patch_size, offset):
        for y in range(0, width - patch_size, offset):
            s, m, soft_m, check = getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height)
            if check:
                r = getIntensityRatio(s, m, soft_m)
                bins = updateBins(bins, binIndx, r)

    r = getFinalRatio(bins, binIndx)
    final_img = fixPatchShadow(shadow_img, hard_mask, soft_mask[..., np.newaxis], r)
    return final_img, bins, r

# ---- Run Shadow Removal ----
shadow_free_image, bins, r = shadowRemover(shadow_img, soft_mask, hard_mask, patch_size=50, offset=10)

# Display input image
plt.figure()
plt.title("Original Shadow Image")
resized_input = cv2.resize(shadow_img, (shadow_img.shape[1] // 4, shadow_img.shape[0] // 4))
plt.imshow(resized_input)
plt.pause(2)

# Display output image
plt.figure()
plt.title("Shadow Free Image")
resized_output = cv2.resize(shadow_free_image, (shadow_free_image.shape[1] // 4, shadow_free_image.shape[0] // 4))
plt.imshow(resized_output)
plt.savefig('Results/' + image, dpi=300)
plt.pause(2)

# Print final RGB Ratios
print('Final RGB Ratios:', r)

# Plot bins
plt.figure()
plt.title('RGB Ratio Space')
plt.xlabel("Bins")
plt.ylabel("# of patches")
plt.plot(bins)
plt.xlim([0, 100])
plt.show()
