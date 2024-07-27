#import numpy as np
#
#def my_imfilter(image, imfilter):
#    """function which imitates the default behavior of the build in scipy.misc.imfilter function.
#
#    Input:
#        image: A 3d array represent the input image.
#        imfilter: The gaussian filter.
#    Output:
#        output: The filtered image.
#    """
#    # =================================================================================
#    # TODO:                                                                           
#    # This function is intended to behave like the scipy.ndimage.filters.correlate    
#    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
#    # of the filter matrix.)                                                          
#    # Your function should work for color images. Simply filter each color            
#    # channel independently.                                                          
#    # Your function should work for filters of any width and height                   
#    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
#    # restriction makes it unambigious which pixel in the filter is the center        
#    # pixel.                                                                          
#    # Boundary handling can be tricky. The filter can't be centered on pixels         
#    # at the image boundary without parts of the filter being out of bounds. You      
#    # should simply recreate the default behavior of scipy.signal.convolve2d --       
#    # pad the input image with zeros, and return a filtered image which matches the   
#    # input resolution. A better approach is to mirror the image content over the     
#    # boundaries for padding.                                                         
#    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
#    # see the desired behavior.                                                       
#    # When you write your actual solution, you can't use the convolution functions    
#    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
#    # Simply loop over all the pixels and do the actual computation.                  
#    # It might be slow.                        
#    
#    # NOTE:                                                                           
#    # Some useful functions:                                                        
#    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
#    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
#    # =================================================================================
#
#    # ============================== Start OF YOUR CODE ===============================
import numpy as np

def my_imfilter(image, imfilter):
    height, width, channels = image.shape
    filter_height, filter_width = imfilter.shape
    
    output = np.zeros_like(image)

    pad_height = int((filter_height - 1) / 2)
    pad_width = int((filter_width - 1) / 2)
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'constant', constant_values=(0, 0))
    print('padded_image', padded_image.shape)

    for i in range(height):
        for j in range(width):
            for ch in range(channels):
                pad = padded_image[i:i+filter_height, j:j+filter_width, ch]
                convolution = np.sum(pad * imfilter)
                output[i, j, ch] = convolution

    return output 
#    # =============================== END OF YOUR CODE ================================
#
#    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
#    # see the desired behavior.
#    # import scipy.ndimage as ndimage
#    # output = np.zeros_like(image)
#    # for ch in range(image.shape[2]):
#    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
#
#    return output


