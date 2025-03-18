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
                patch = image[:, i*self.filter_size:(i+1)*self.filter_size,
                              j*self.filter_size:(j+1)*self.filter_size]
                yield patch, i, j

    def forward(self, image):
        channels, height, width = image.shape
        self.output = np.zeros((channels, height // self.filter_size, width // self.filter_size))
        
        for patch, i, j in self.image_region(image):
            self.output[:, i, j] = np.max(patch, axis=(1, 2))
        return self.output

    def backward(self, dvalues):
        dinputs = np.zeros(self.image.shape)
        
        for patch, i, j in self.image_region(self.image):
            for c in range(patch.shape[0]):
                max_val = np.max(patch[c])
                for h in range(self.filter_size):
                    for w in range(self.filter_size):
                        if patch[c, h, w] == max_val:
                            dinputs[c, i*self.filter_size+h, j*self.filter_size+w] = dvalues[c, i, j]
        return dinputs
            

