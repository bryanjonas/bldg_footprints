import os
import rasterio as rio
import torch
from torch.nn import functional as F
from skimage.io import imsave
import cv2
from model import *
import math
import segmentation_models_pytorch as smp
import numpy as np

#device = torch.cuda.set_device('cuda:1')

def main():
    device = torch.device('cuda:0')
    config = vars(parse_args())

    #Build model
    model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            )

    #Load into GPU
    model = model.to(device)
    
    #Load save model
    model.load_state_dict(torch.load(config['saved_model']))
    
    network_size = (config['input_w'],
                        config['input_h'],
                        config['input_channels'])
    
    src = rio.open(config['image'])

    img_shape = src.read().shape

    if img_shape[0] > 1: #Non-grayscale 
        if config['input_channels'] == 3:
            #C,W,H to W,H,C
            img_array = src.read((1,2,3)).transpose(1,2,0)
        
        elif config['input_channels'] == 1:
            img_array = cv2.cvtColor(src.read((1,2,3)).transpose(1,2,0),cv2.COLOR_RGB2GRAY)[..., np.newaxis]
            
    else:
        if config['input_channels'] == 1:
            img_array = src.read(1)[..., np.newaxis]
        
        elif config['input_channels'] == 3:
            img_array = cv2.cvtColor(src.read(1), cv2.COLOR_GRAY2RGB)
    
    #OLD CODE
    #Note: This scaling is probably what you want to use in practice.
    #if img_array.dtype == 'uint16':
    #   img_array = (img_array / 2**16) * 255
    
    #Note: This scaling is just to reduce the color depth to 8-bit for SpaceNet images (11-bit)
    #img_array = (img_array / 2**11) * 255
    ##

    img_array = min_max_scale(img_array, channels = 1)
    
    image_size = list(img_array.shape[0:2]) #Get image size and push to a list
    tile_size = list(network_size[0:2]) #Get tile size as first two dimensions of network_size
        
    corners = find_corners(image_size, tile_size, network_size, config['overlap'])
    tiles_array = create_tiles(img_array, corners, network_size)

    #Necessary for prepping array for PyTorch
    tiles_array = tiles_array.astype('float32')

    tiles_array = tiles_array.transpose(0, 3, 1, 2)
    tiles_array = torch.from_numpy(tiles_array)
    tiles_array = tiles_array.to(device)
    
    #Inference
    with torch.no_grad():
        output = model(tiles_array)
        output = torch.sigmoid(output)
        #output = F.relu(output).cpu().numpy()
        output = output.cpu().numpy()
        
    torch.cuda.empty_cache()
    
    #final_tiles = np.empty((output.shape[0], 1, output.shape[2], output.shape[3]))
    #for tile in range(0, output.shape[0]):
    #    #"Stretch" tile to one long array (W x H) of sets of three values
    #    stretch_tile = output[tile,:,:,:].transpose(1,2,0).reshape(output.shape[2]*output.shape[3], output.shape[1])
    #    #Apply logistic regression classifier
    #    reg_tile = np.apply_along_axis(reg_func, 1, stretch_tile)
    #    #Reshape and add it to final tiles array
    #    final_tiles[tile,:,:,:] = reg_tile.reshape(output.shape[2], output.shape[3], 1).transpose(2, 0, 1)
    
    final_tiles = output

    #For testing: save out each individual tile
    #for tile in range(0, final_tiles.shape[0]):
    #    imsave(str(tile) + '.png', final_tiles[tile,:,:,:].transpose(1, 2, 0))

    final_assem = assemble_pred(final_tiles[:,0,:,:], corners, image_size)
    
    final_assem = (final_assem > 0.5).astype(int)

    final_assem = np.array(final_assem, dtype=np.uint8)
    
    if config['png']:
    #For testing: save final prediction image as PNG
        imsave('final.png', final_assem)
    
    output_path = os.path.join(config['output_dir'],config['image'].split('/')[-1])

    
    #Use RIO to save output with attributes from original geotiff
    with rio.open(
        output_path,
        'w',
        driver='GTiff',
        height=final_assem.shape[0],
        width=final_assem.shape[1],
        count=1,
        dtype=final_assem.dtype,
        crs=src.crs,
        nodata='0',
        transform=src.transform,
    ) as dst:
        dst.write(final_assem, 1)

if __name__ == '__main__':
    main()