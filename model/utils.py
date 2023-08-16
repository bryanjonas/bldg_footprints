from torch.utils.data import Dataset, IterableDataset
import numpy as np
import argparse
import math

def min_max_scale(x, channels):
        import numpy as np
        means = np.array([0]*channels)
        maxes = np.array([np.max(x)]*channels)
        stds = np.array([1]*channels)
        x = (x - (means * maxes) + (1e-8)) / (stds * maxes + (1e-8))
        return x

def find_corners(image_size, tile_size, network_size, overlap=0):
    # Function that finds the coordinates of the corners of the tiles
    # Inputs: image_size - 2D array
    #         tile_size - 2D array
    #         network_size - 3D array
    #         overlap - scalar value
    # Output: List of 3D arrays -
    #           (coord of top left corner, coord of bottom right corner, size of that tile)
    #

    #Set up variables for iteration
    topCorner = [0,0]
    shift = False
    corners = []
    #Easily track incomplete squares
    size = [network_size[0], network_size[1]]

    while True:
        while not shift:

            bottomCorner = list(map(lambda x,y:x+y, topCorner, tile_size)) #Make a full tile
    
            if bottomCorner[0] > image_size[0]: #If the full tile bottom is off the image
                bottomCorner[0] = image_size[0] #Stop to the tile bottom at the image edge
        
            if bottomCorner[1] > image_size[1]: #If the right egde of the tile is off the image
                bottomCorner[1] = image_size[1] #Stop the right edge of the tile at the image edge
                shift = True #Time to shift to the next y position
        
            size = list(map(lambda x,y: x-y, bottomCorner, topCorner)) #Record size of actual image in this tile
    
            corners += [np.array([topCorner, bottomCorner, size])] #Add this into to our list of corners
    
            topCorner[1] += tile_size[1] - overlap #Move top corner x-position over
            
        #Shift to next y-position
        shift = False #Reset flag
            
        topCorner[0] += tile_size[0] - overlap #Change top corner y-position
    
        topCorner[1] = 0 #Change top corner x-position
    
        if bottomCorner == image_size: #Check to see if we are at the very bottom corner
            break #Stop if we are
    return corners

def create_tiles(image, corners, network_size):
    # Function that uses the corners list to tile the image.
    # Inputs: image - 3D array
    #         corners - list
    #         network_size - 3D array
    # Output: 3D array
    #
    
    tiles_list = [] #List to hold tile arrays
    for cornerSet in corners:
        topY = cornerSet[0,0]
        topX = cornerSet[0,1]
        botY = cornerSet[1,0]
        botX = cornerSet[1,1]
        
        tile = image[topY:botY, topX:botX, :]
        if list(tile.shape) != network_size: #Need to ensure tiles are all the proper size for the network
            inter_array = np.zeros((np.array(network_size)))
            inter_array[0:tile.shape[0],0:tile.shape[1],:] = tile #Pad out the tile size with zeros
            tile = inter_array
        tiles_list += [tile]

    tilesArr = np.array(tiles_list)
        
    return tilesArr

def assemble_pred(predArr, corners, image_size):
    # Function to assemble final image from prediction array.
    # Inputs: predArr - 3D array (tiles, width, height)
    #         corners - list
    #         image_size - 3D array
    # Output: 3D array
    #
    
    final_pred = np.zeros((image_size[0], image_size[1]))
    final_pred[:,:] = np.nan #NaN so we only average the overlap
    corn_idx = 0
    for pred in predArr:
        cornerSet = corners[corn_idx]
        topY = cornerSet[0,0]
        topX = cornerSet[0,1]
        botY = cornerSet[1,0]
        botX = cornerSet[1,1]
        tile_shape = cornerSet[2]

        img_tile = final_pred[topY:botY, topX:botX] #Current values in tile from our final prediction image
        
        #Deal with overlap
        flat_img_tile = np.ndarray.flatten(img_tile) #Flatten to a 1D array of values
        pred = pred[0:tile_shape[0],0:tile_shape[1]] #Get rid of padding
        flat_pred_tile = np.ndarray.flatten(pred) #Flatten to 1D array of values
        new_img_tile = list(map(lambda x, y: y if np.isnan(x) else (x+y)/2, flat_img_tile, flat_pred_tile)) #Average the predictions if x isn't NaN
        new_img_tile = np.asarray(new_img_tile).reshape(tile_shape) #list to array
        final_pred[topY:botY, topX:botX] = new_img_tile #Replace part of image with new tile
            
        corn_idx += 1  
    return final_pred    

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--saved_model', default=None,
                        help='Saved model (*.pth) location',
                        required=True)
    
    parser.add_argument('--input_w', default=256, type=int,
                        help='Width of network image input',
                        required=True)
    
    parser.add_argument('--input_h', default=256, type=int,
                        help='Height of network image input',
                        required=True)
    
    parser.add_argument('--input_channels', default=1, type=int,
                       help='Number of channels in input image',
                       required=True)
    
    parser.add_argument('--num_classes', default=1, type=int, 
                        help='Number of layers in network output',
                        required=True)
    
    parser.add_argument('--overlap', default=0, type=int,
                        help='Number of pixels to overlap during inference',
                        required=True)
    
    parser.add_argument('--image', 
                       help='Image for inference',
                       required=True)

    parser.add_argument('--output_dir',
                        help='Output directory',
                        required=True)

    parser.add_argument('--png', type=bool, default=False,
                        help='Output final.png', required=True)
    
    config = parser.parse_args()

    return config

def load_testset(json_path='../test_set.json', device=None, img_size=256, transform=None, mask_dir="mask", img_dir="3band"):
    import json
    import rasterio as rio
    import numpy as np
    import pathlib
    import albumentations as A
    import skimage.io as io
    import os
    import torch
    from tqdm import tqdm

    with open(json_path) as file:
        test_set_list = json.load(file)

    batch_size = len(test_set_list)

    x = torch.empty((batch_size, 3, img_size, img_size)).to(device)
    y = torch.empty((batch_size, 1, img_size, img_size)).to(device)

    for idx, img_path in enumerate(tqdm(test_set_list)):
        if pathlib.Path(img_path).suffix == '.jpg':
            img = io.imread(img_path).astype(np.float32)     
        else:
            img = rio.open(img_path).read((1,2,3)).astype(np.float32)
            img = img.transpose(1,2,0)

        mask_root = str(img_path).split(img_dir, maxsplit=1)[0]
        if mask_root.find('crowdai') == -1:
            mask_file = 'AOI' + str(img_path).split(img_dir, maxsplit=1)[1].split('AOI')[-1].split('.tif')[0] + '_mask.npy'
        else:
            mask_file = str(img_path).split('/')[-1].split('.')[0] + '.npy'
        y_mask = np.load(os.path.join(mask_root, mask_dir, mask_file))    
        y_mask = np.expand_dims(y_mask, axis=2)

        if transform:
            transformed = transform(image=img, mask=y_mask)
            img = transformed['image']
            y_mask = transformed['mask']

        means = np.array([0]*3)
        maxes = np.array([np.max(img)]*3)
        stds = np.array([1]*3)
        img = (img - (means * maxes) + (1e-8)) / (stds * maxes + (1e-8))

        img = img.transpose(2,0,1)
        y_mask = y_mask.transpose(2,0,1)

        y_mask = torch.from_numpy(y_mask)
        img = torch.from_numpy(img)
        img = img.to(device)
        y_mask = y_mask.to(device)

        x[idx,:,:,:] = img
        y[idx,:,:,:] = y_mask

    return x, y
    

class BetaNetDataset(Dataset):
    def __init__(self, device, paths, img_dir, mask_dir, train_val = "train", img_size = 256, use_pan = True, batch_size = 16, zero_prop = None, transform=None, ksize = 7, sigma=3, constant=200, lam=0.05):
        super(Dataset).__init__()
        import os
        import pathlib
        import random
        import math
        import rasterio as rio
        import numpy as np
        import skimage.io as io
        import json

        self.use_pan = use_pan
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.train_val = train_val
        self.channels = [1,2,3]
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.zero_prop = zero_prop
        self.transform = transform
        self.ksize = ksize
        self.sigma = sigma
        self.constant = constant
        self.lam = lam


        img_paths = list(os.path.join(path, train_val, img_dir) for path in paths)
        self.img_list = []
        for path in img_paths:
            curr_list = list(pathlib.Path(path).iterdir())
            self.img_list.extend(curr_list)

        #Drop images what have greater than zero_prop of empty pixels
        if self.zero_prop:
            pop_list = []
            with open('img_prop_records.json','r') as file:
                prop_dict = json.load(file)
            for idx, img in enumerate(self.img_list):
                if prop_dict[str(img)] > (self.zero_prop/100):
                    pop_list.append(idx)
            
            for i in sorted(range(len(pop_list)), reverse=True):
                del self.img_list[i]

        self.gen_random_list()
       
    def gen_random_list(self):
        import random

        self.rimg_list = random.sample(range(len(self.img_list)), len(self.img_list))    


    def get_mask_from_path(self, img_path):
        import numpy as np
        import os
        
        mask_root = str(img_path).split(self.img_dir, maxsplit=1)[0]

        if mask_root.find('crowdai') == -1:
            mask_file = 'AOI' + str(img_path).split(self.img_dir, maxsplit=1)[1].split('AOI')[-1].split('.tif')[0] + '_mask.npy'
        else:
            mask_file = str(img_path).split('/')[-1].split('.')[0] + '.npy'
        mask_path = os.path.join(mask_root, self.mask_dir, mask_file)    

            
        return np.load(mask_path)

    def get_edges(self, y_mask_inner):
        import skimage as ski
        import numpy as np
        
        y_mask_edge = np.zeros_like(y_mask_inner)

        padded_mask_inner = np.pad(y_mask_inner, pad_width=1, mode='constant', constant_values=0)
        contours = ski.measure.find_contours(padded_mask_inner, 0.5)
        for contour in contours:
            for point in contour:
                point[0] -= 1
                point[1] -= 1
                if point[0] < 0:
                    point[0] = 0
                if point[1] < 0:
                    point[0] = 0
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))

            rr, cc = ski.draw.polygon_perimeter(contour[:,0], contour[:,1], y_mask_inner.shape)
            y_mask_edge[rr, cc] = 1
        
        #return ski.morphology.dilation(y_mask_edge)
        return y_mask_edge

    def random_repeat_crop(self, x, mask_inner):
        import random

        top_left_x = random.randrange(0, mask_inner.shape[0]-self.img_size)
        top_left_y = random.randrange(0, mask_inner.shape[1]-self.img_size)

        new_mask = mask_inner[top_left_x:top_left_x+self.img_size, top_left_y:top_left_y+self.img_size]
        new_x = x[:,top_left_x:top_left_x+self.img_size, top_left_y:top_left_y+self.img_size]

        return new_x, new_mask
    
    def create_weight_map(self, y_mask_inner, y_mask_edge):
        import cv2
        import numpy as np

        y_mask_edge_cp = np.copy(y_mask_edge).astype(float)
        y_mask_weight = cv2.GaussianBlur(y_mask_edge_cp, (self.ksize, self.ksize), self.sigma) / self.constant
        y_mask_weight = y_mask_weight - y_mask_inner

        #Lowest value is zero
        y_mask_weight = np.where(y_mask_weight<0, 0, y_mask_weight)

        return y_mask_weight

    def min_max_scale(self, x):
        import numpy as np
        means = np.array([0]*len(self.channels))
        maxes = np.array([np.max(x)]*len(self.channels))
        stds = np.array([1]*len(self.channels))
        x = (x - (means * maxes) + (1e-8)) / (stds * maxes + (1e-8))
        return x

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        import rasterio as rio
        import numpy as np
        import torch
        import torchvision
        import time
        import albumentations as A
        import cv2
        import skimage.io as io
        import pathlib
        from PIL import Image

        if pathlib.Path(self.img_list[index]).suffix == '.jpg':
            x = io.imread(self.img_list[index]).astype(np.float32)     
        else:
            x = rio.open(self.img_list[index]).read(self.channels).astype(np.float32)
            x = x.transpose(1,2,0)

        if self.use_pan:
            #Convert to 8-bit
            if x.max() > 255:
                x = np.asarray((x / 2**11) * 2**8)
            x = x.astype(np.uint8)
            x_img = Image.fromarray(x)
            g_x_ing = x_img.convert('L')
            x = np.array(g_x_ing)
            x = np.expand_dims(x, axis=2)

        


        y_mask_inner = self.get_mask_from_path(self.img_list[index])
        y_mask_inner = np.expand_dims(y_mask_inner, axis=2)

        if self.transform:
            transformed = self.transform(image=x, mask=y_mask_inner)
            x = transformed['image']
            y_mask_inner = transformed['mask']

        y_mask_inner = y_mask_inner.squeeze()

        #Get mask of cropped image
        y_mask_edge = self.get_edges(y_mask_inner)

        #Get weight map
        y_mask_weight = self.create_weight_map(y_mask_inner, y_mask_edge)

        y = np.stack((y_mask_inner, y_mask_weight))

        #y = torch.from_numpy(y)
        #x = torch.from_numpy(x)
        #x = x.to(self.device)
        #y = y.to(self.device)

        return x, y

    def gen_batch(self):
        import torch
        import numpy as np
        
        if len(self.rimg_list) < (self.batch_size * 2):
            self.gen_random_list()

        #Track batch index
        j = 0

        if self.use_pan:
            chan_dim = 1
        else:
            chan_dim = len(self.channels)

        x_batch = torch.empty((self.batch_size, chan_dim, self.img_size, self.img_size)).to(self.device)
        y_batch = torch.empty((self.batch_size, 4, 2, self.img_size, self.img_size)).to(self.device)
        for i in range(0, self.batch_size * 2, 2):
            lam_i = np.random.uniform(0, self.lam)
            #Values comes out of getitem are shaped list:
            # x - [i, m, n]
            #   i : channel
            #   m : width
            #   n : height
            # y - [i, m, n]
            #   i : 
            #     0 - building mask
            #     1 - weight map
            #   m : width
            #   n : height

            x1, y1 = self.__getitem__(self.rimg_list[i])
            x2, y2 = self.__getitem__(self.rimg_list[i+1])

            x3 = (x1 * lam_i) + (x2 * (1 - lam_i))
            x3 = min_max_scale(x3, chan_dim)
            x3 = x3.transpose(2,0,1)
            x3 = torch.from_numpy(x3)
            x3 = x3.to(self.device)
            x_batch[j,:,:,:] = x3

            #Mixup ground truth and weight map
            y3 = (y1 * lam_i) + (y2 * (1 - lam_i))

            y1 = torch.from_numpy(y1)
            y2 = torch.from_numpy(y2)
            y3 = torch.from_numpy(y3)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            
            y_batch[j,0,:,:,:] = y1
            y_batch[j,1,:,:,:] = y2
            y_batch[j,2,:,:,:] = y3
            y_batch[j,3,:,:,:] = lam_i

            #Increment batch index
            j += 1

        for i in sorted(range(self.batch_size * 2), reverse=True):
            del self.rimg_list[i]
        
        return x_batch, y_batch

    def gen_raw_x(self):
        import torch
        import numpy as np
        import rasterio as rio
        
        if len(self.rimg_list) < self.batch_size:
            self.gen_random_list()
        
        x_batch = []

        for i in range(self.batch_size):
            path = self.img_list[self.rimg_list[i]]
            x = rio.open(path).read(self.channels).astype(np.float32)
            x_batch.append(torch.from_numpy(x))

        for i in sorted(range(self.batch_size), reverse=True):
            del self.rimg_list[i]
        
        return x_batch
