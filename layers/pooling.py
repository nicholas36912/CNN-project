import numpy as np

class maxpool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

        def image_region(self, image):
            self.image=image
            channels, height, width = image.shape
            new_height = height // self.filer_size
            new_width = width // self.filter_size

            for i in range(new_height):
                for j in range(new_width):
                    patch = image[:,i*self.filer_size:(i+i)*self.filter_size,
                                  j*self.filter_size(j+j)*self.filter_size]
                    yield patch, i, j

        def forward (self, image):
            channels, height, width = image.shape
            self.output = np.zeros((channels, height // self.filter_size, width // self.filter_size))

            for patch, i, j in self.image_region(image):
                self.output[:,i,j] = np.max(patch, axis = (1,2))
                return self.output
            

