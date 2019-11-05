
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################
    
    # Get image size
    H = img.shape[0]
    W = img.shape[1]

    # Initialize height and width index
    # Indicating which color should be kept.
    # Ex. R1C1: G, R1C2: R, R2C1:B, R2C2:G
    H_idx = 0
    W_idx = 0

    # Create the H*W output numpy array.
    output = np.zeros((H, W))

    for i in range(H):
        if H_idx == 0:
            H_idx += 1
            for j in range(W):
                if W_idx == 0:
                    W_idx += 1
                    if(pattern == 'GRBG'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'RGGB'):
                        output[i][j] = img[i][j][0]
                    elif(pattern == 'GBRG'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'BGGR'):
                        output[i][j] = img[i][j][2]
                else:
                    W_idx -= 1
                    if(pattern == 'GRBG'):
                        output[i][j] = img[i][j][0]
                    elif(pattern == 'RGGB'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'GBRG'):
                        output[i][j] = img[i][j][2]
                    elif(pattern == 'BGGR'):
                        output[i][j] = img[i][j][1]
        else:
            H_idx -= 1
            for j in range(W):
                if W_idx == 0:
                    W_idx += 1
                    if(pattern == 'GRBG'):
                        output[i][j] = img[i][j][2]
                    elif(pattern == 'RGGB'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'GBRG'):
                        output[i][j] = img[i][j][0]
                    elif(pattern == 'BGGR'):
                        output[i][j] = img[i][j][1]
                else:
                    W_idx -= 1
                    if(pattern == 'GRBG'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'RGGB'):
                        output[i][j] = img[i][j][2]
                    elif(pattern == 'GBRG'):
                        output[i][j] = img[i][j][1]
                    elif(pattern == 'BGGR'):
                        output[i][j] = img[i][j][0]

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

