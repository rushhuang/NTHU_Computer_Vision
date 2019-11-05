
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    
    # Get image size
    H = img.shape[0]
    W = img.shape[1]

    # Initialize height and width index
    # Indicating which color should be kept.
    # Ex. R1C1: G, R1C2: R, R2C1:B, R2C2:G
    H_idx = 0
    W_idx = 0

    # Create the H*W mask numpy array.
    mask = np.zeros((H, W))
    
    for i in range(H):
        if H_idx == 0:
            H_idx += 1
            for j in range(W):
                if W_idx == 0:
                    W_idx += 1
                    if(pattern == 'GRBG'):
                        mask[i][j] = 1
                    elif(pattern == 'RGGB'):
                        mask[i][j] = fr
                    elif(pattern == 'GBRG'):
                        mask[i][j] = 1
                    elif(pattern == 'BGGR'):
                        mask[i][j] = fb
                else:
                    W_idx -= 1
                    if(pattern == 'GRBG'):
                        mask[i][j] = fr
                    elif(pattern == 'RGGB'):
                        mask[i][j] = 1
                    elif(pattern == 'GBRG'):
                        mask[i][j] = fb
                    elif(pattern == 'BGGR'):
                        mask[i][j] = 1
        else:
            H_idx -= 1
            for j in range(W):
                if W_idx == 0:
                    W_idx += 1
                    if(pattern == 'GRBG'):
                        mask[i][j] = fb
                    elif(pattern == 'RGGB'):
                        mask[i][j] = 1
                    elif(pattern == 'GBRG'):
                        mask[i][j] = fr
                    elif(pattern == 'BGGR'):
                        mask[i][j] = 1
                else:
                    W_idx -= 1
                    if(pattern == 'GRBG'):
                        mask[i][j] = 1
                    elif(pattern == 'RGGB'):
                        mask[i][j] = fb
                    elif(pattern == 'GBRG'):
                        mask[i][j] = 1
                    elif(pattern == 'BGGR'):
                        mask[i][j] = fr

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask