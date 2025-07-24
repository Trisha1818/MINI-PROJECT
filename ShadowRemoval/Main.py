import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

import closed_form_matting


# image = 'road_shadow.png'
# image = 'crossing.png'

# image = 'road2.png'

image = 'grass.png'
image = 'nail.png'
image = 'lawn.png'

image = 'z10.png'
image = 'snow.png'

# image = 'tennis.png'

# image = 'snow.png'
# image = 'footpath.png'



# Load shadow image from file first
image = 'snow.png'

# Load shadow image
shadow_img = cv2.cvtColor(cv2.imread('Samples/ShadowImages/' + image), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
shadow_img = cv2.resize(shadow_img, (shadow_img.shape[1] // 2, shadow_img.shape[0] // 2))
plt.figure()
plt.imshow(shadow_img)
plt.pause(3)

# Load hard mask
# Load hard mask
hard_mask_path = 'Samples/HardMasks/' + image
hard_mask_raw = cv2.imread(hard_mask_path)

if hard_mask_raw is None:
    raise FileNotFoundError(f"Could not load hard mask image: {hard_mask_path}")

hard_mask = cv2.cvtColor(hard_mask_raw, cv2.COLOR_BGR2RGB).astype('double') / 255.0
hard_mask = cv2.resize(hard_mask, (hard_mask.shape[1] // 2, hard_mask.shape[0] // 2))

plt.imshow(hard_mask)
plt.pause(3)

# Load scribbles
scribbles = cv2.imread('Samples/Scribbles/' + image, cv2.IMREAD_COLOR) / 255.0
scribbles = cv2.resize(scribbles, (scribbles.shape[1] // 2, scribbles.shape[0] // 2))

# Apply closed-form matting
soft_mask = closed_form_matting.closed_form_matting_with_scribbles(shadow_img, scribbles)

# Skip disk writing and directly invert
soft_mask = 1 - soft_mask
plt.imshow(soft_mask)
plt.pause(3)


#shadow removal functions
def computeAverage(arr):
    arr[arr == 0] = np.nan
    arr_mean = np.nanmean(arr, axis=1)
    arr_mean = np.nanmean(arr_mean, axis=0)

    return arr_mean

def getIntensityRatio(imgPatch, maskPatch, softMaskPatch):
    s = imgPatch
    m = maskPatch

    shd = s * m
    non_shd = s * np.logical_not(m)

    shd_mean = computeAverage(shd)
    non_shd_mean = computeAverage(non_shd)

    shd_K = softMaskPatch * m
    non_shd_K = softMaskPatch * np.logical_not(m)

    shd_mean_K = computeAverage(shd_K)
    non_shd_mean_K = computeAverage(non_shd_K)

    denominator = (shd_mean * non_shd_mean_K) - (non_shd_mean * shd_mean_K)

    # Avoid divide-by-zero or tiny values
    denominator = np.where(np.abs(denominator) < 1e-5, 1e-5, denominator)

    r = (non_shd_mean - shd_mean) / denominator

    return r



def getMaskRatio(m):
    
    totalCount = m.shape[0]*m.shape[1]
    
    
    mOnes = np.count_nonzero(m[:,:,0]) 
    mZeros = totalCount - mOnes
    
    
    patchRatio = mOnes  / (mOnes + mZeros)
    
    return patchRatio



def getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height):
    xE = x + patch_size
    yE = y + patch_size

    if xE >= height:
        xE = height
    if yE >= width:
        yE = width

    soft_m = soft_mask[x:xE, y:yE].copy()  # Still 2D
    soft_m = np.repeat(soft_m[:, :, np.newaxis], 3, axis=2)  # Convert to 3D shape (H, W, 3)
  # ✅ 2D patch
    print("soft_mask shape:", soft_mask.shape)



    m = hard_mask[x:xE, y:yE, :].copy()
    s = shadow_img[x:xE, y:yE, :].copy()

    maskRatio = getMaskRatio(m)

    if 0.49 < maskRatio < 0.51:
        return s, m, soft_m, True
    else:
        return None, None, None, False


    
def updateBins(bins, binIndx, r):
    
    for i in range(3):
        currentR = round(r[i], 1)
        
        if(0 <= int(currentR*10) and int(currentR*10) < len(binIndx)):
            bins[int(currentR*10), i]  = bins[int(currentR*10),i] + 1
        
    return bins


def getFinalRatio(bins, binIndx):
    
    r = np.zeros(3).astype('float64')

    indx = np.where(bins == np.max(bins, axis=0))
    
    prevY = 0
    prevYCount = 0
    
    for x, y in zip(indx[0], indx[1]):
        
        r[y] = r[y] + binIndx[x].astype('float64')
        
        r = np.nan_to_num(r)
        
        if(y != prevY):
            r[prevY]  = r[prevY]/prevYCount
            prevYCount = 0
            
        prevY  = y
        prevYCount = prevYCount + 1
    
    r[prevY]  = r[prevY]/prevYCount
    
    return r
        
    
def removeShadowImage(shadow_img, hard_mask, softMaskPatch, ratio):
    
    shadow_mapper = (ratio + 1)/(ratio*softMaskPatch + 1)
    
    
    mapped_image = np.multiply(shadow_img, shadow_mapper)

    max_vals = np.zeros(3, dtype= 'float64')
    max_valsBG = np.zeros(3, dtype= 'float64')


    fg = (hard_mask*mapped_image)
    
    bg = shadow_img*np.logical_not(hard_mask)


    max_vals[0]  = np.amax(fg[:,:,0])
    max_vals[1]  = np.amax(fg[:,:,1])
    max_vals[2]  = np.amax(fg[:,:,2])
    
    max_valsBG[0]  = np.amax(bg[:,:,0])
    max_valsBG[1]  = np.amax(bg[:,:,1])
    max_valsBG[2]  = np.amax(bg[:,:,2])

    shadow_free_image = (fg/max_vals) + (shadow_img*np.logical_not(hard_mask))
    
    return shadow_free_image



def fixPatchShadow(imgPatch, maskPatch, softMaskPatch, ratio):
    # Expand softMaskPatch to 3 channels
    if len(softMaskPatch.shape) == 2:
        softMaskPatch = np.repeat(softMaskPatch[:, :, np.newaxis], 3, axis=2)

    # Now compute mapper safely
    mapper = (ratio + 1) / (ratio * softMaskPatch + 1e-5)

    fixed = imgPatch * mapper

    fg = fixed * maskPatch
    bg = fixed * (1 - maskPatch)

    final = fg + bg
    return final



def rescale_images_linear(le):
    '''
    Helper function to rescale images in visible range
    '''
    le_min = le[le != -float('inf')].min()
    le_max = le[le != float('inf')].max()
    le[le==float('inf')] = le_max
    le[le==-float('inf')] = le_min

    le = (le - le_min) / (le_max - le_min)

    return le


def shadowRemover(shadow_img, soft_mask, hard_mask, patch_size = 12, offset = 1):


    height = shadow_img.shape[0]
    width = shadow_img.shape[1]


    binIndx = np.arange(0,100,0.1)
    bins = np.zeros((len(binIndx),3))



    for x in np.arange(0,height,offset)[0:-1]:
        for y in np.arange(0,width,offset)[0:-1]:


            s, m, soft_m, check = getPatch(soft_mask, hard_mask, shadow_img, x, y, patch_size, width, height)

            if(check):

                r = getIntensityRatio(s, m, soft_m)

                bins = updateBins(bins, binIndx, r)



    r = getFinalRatio(bins, binIndx)

    shadow_free_image = fixPatchShadow(shadow_img.copy(), hard_mask.copy(), soft_mask.copy(), r)
    
    
    return shadow_free_image, bins, r




shadow_free_image, bins, r = shadowRemover(shadow_img, soft_mask, hard_mask, patch_size=50, offset=10)

plt.figure()
resized = cv2.resize(shadow_img, (shadow_img.shape[1] // 4, shadow_img.shape[0] // 4))
plt.imshow(resized)

plt.pause(3)

plt.figure()
resized = cv2.resize(shadow_free_image, (shadow_free_image.shape[1] // 4, shadow_free_image.shape[0] // 4))
plt.imshow(resized)

plt.savefig('Results/' + image, dpi = 300)
plt.pause(3)


print('Final RGB Ratios:', r)

plt.figure()
plt.title('RGB Ratio Space')
plt.xlabel("Bins")
plt.ylabel("# of patches")
plt.plot(bins)
plt.xlim([0, 100])  
