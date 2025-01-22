import numpy as np
from formulas import activations

class convolve3d:

    def __init__(self, num_filters, filter_size, channels):
    #initialize params
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.channels = channels
    #define the kernel
        self.kernel = np.random.randn(num_filters, filter_size, filter_size, channels)

    #extract image region with generator function
    def image_region(self, image):
        #initialize params
        height, width, channels = image.shape
        self.image = image
        #compute height and width of output after applying kernel
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        #loop to iterate over rows of image where filter is applied
        for i in range(output_height):
            #nested loop to iterate over columns of the image where filter is applied
            for j in range (output_width):
                #extract image patch of kernel starting at i, j 
                image_patch = image[i:i+self.filter_size, j:j+self.filter_size]
                #yeild 1 patch at once
                yield image_patch

    def forward(self, image):
        height, width, channels = image.shape
        self.image - image 
        #initialize output matrix
        self.output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        
        for image_patch, i, j in self.image_region(image):
            for f in range (self.num_filters):
                self.output[i,j,f] = np.sum (image_patch * self.kernel[f])
        self.output = activations.relu(self.output)

    

