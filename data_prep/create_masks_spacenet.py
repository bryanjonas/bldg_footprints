#/usr/bin/python3 /workspaces/BetaNet/data_prep/create_masks.py -s /data/spacenet/AOI_1_Rio/summarydata/AOI_1_RIO_polygons_solution_3band.csv -t /data/spacenet/AOI_1_Rio/3band -o /data/spacenet/AOI_1_Rio/masks/ 

#/usr/bin/python3 /workspaces/BetaNet/data_prep/create_masks.py -s /data/spacenet/AOI_2_Vegas/summarydata/AOI_2_Vegas_Train_Building_Solutions.csv -t /data/spacenet/AOI_2_Vegas/3band -o /data/spacenet/AOI_2_Vegas/masks/ 
#/usr/bin/python3 /workspaces/BetaNet/data_prep/create_masks.py -s /data/spacenet/AOI_3_Paris/summarydata/AOI_3_Paris_Train_Building_Solutions.csv -t /data/spacenet/AOI_3_Paris/3band -o /data/spacenet/AOI_3_Paris/masks/ 
#/usr/bin/python3 /workspaces/BetaNet/data_prep/create_masks.py -s /data/spacenet/AOI_4_Shanghai/summarydata/AOI_4_Shanghai_Train_Building_Solutions.csv -t /data/spacenet/AOI_4_Shanghai/3band -o /data/spacenet/AOI_4_Shanghai/masks/ 
#/usr/bin/python3 /workspaces/BetaNet/data_prep/create_masks.py -s /data/spacenet/AOI_5_Khartoum/summarydata/AOI_5_Khartoum_Train_Building_Solutions.csv -t /data/spacenet/AOI_5_Khartoum/3band -o /data/spacenet/AOI_5_Khartoum/masks/ 

def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Utility to create masks.')
    #Commented out for now. May need to add functionality back to handle AOI 6-7
    #parser.add_argument('-a',
    #                    #required=True,
    #                    help='specify AOI for mask creation.',
    #                    dest='aoi',
    #                    type=str,
    #                    choices=('AOI_1', 'AOI_2', 'AOI_3', 'AOI_4', 'AOI_5', 'AOI_6', 'AOI_7'))
    parser.add_argument('-s', '--summary',
                        metavar='/data/spacenet/...',
                        dest='summary',
                        #required=True,
                        help='specify full path for AOI summary CSV',
                        type=str)
    parser.add_argument('-t', '--tiff',
                        #required=True,
                        dest='tiff',
                        metavar='/data/spacenet/...',
                        help='specify full path for AOI tif images',
                        type=str)
    parser.add_argument('-o', '--output',
                        #required=True,
                        dest='output',
                        metavar='/data/spacenet/...',
                        help='specify full path for output of mask image files',
                        type=str)
    
    return parser

def parse_aoi(args):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import os
    from tqdm import tqdm
    import rasterio as rio
    import skimage as ski
    import re

    #Regex filter for polygon points
    numbers = re.compile(r'\d+(?:\.\d+)?')

    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    summary_csv_df = pd.read_csv(args.summary)
    
    #Only search unique image IDs to save time
    image_ids_arr = summary_csv_df['ImageId'].unique()
    
    ###DEBUGGING
    #image_ids_arr = image_ids_arr[0:5]

    image_lookup_dict = {}

    filenames = os.listdir(args.tiff)

    print('Finding dimensions of each tiff...')
    for image_id in tqdm(image_ids_arr):
        #Open TIFF to get size
        file_flag = False

        for file in filenames:
            if file.find(image_id + '.') != -1:
                raster = rio.open(os.path.join(args.tiff, file))
                tiff_h, tiff_w = raster.height, raster.width
                file_flag = True
                raster.close()
                break
        if file_flag:
            image_lookup_dict[image_id] = ((tiff_h, tiff_w))
        else:
            image_lookup_dict[image_id] = ((0, 0))
    
    print('Gathering polygons for each tiff...')
    image_polygons_dict = {}
    for image_id in tqdm(image_ids_arr):
        curr_id_rows = summary_csv_df[summary_csv_df['ImageId'] == image_id]
        image_polygons_dict[image_id] = []
        for idx, row in curr_id_rows.iterrows():
            image_polygons_dict[image_id] += [row['PolygonWKT_Pix']]

    print('Building masks for each tiff...')
    for image_id in tqdm(image_ids_arr):
        image_dim = image_lookup_dict[row['ImageId']]
        if image_dim == (0, 0):
            #No matching tiff file
            break

        curr_id_polys = image_polygons_dict[image_id]
        masked_arr = np.zeros(image_dim)
        try:
            for poly in curr_id_polys:
                if poly.find('EMPTY') != -1:
                    polygon_arr = np.empty(image_dim)
                else:
                    polygon_arr = np.array(list(numbers.findall(i) for i in poly.split("((")[1].split("))")[0].split(','))).astype(float)

                y_arr = polygon_arr[:,0:1].flatten()
                x_arr = polygon_arr[:,1:2].flatten()
            
                rr, cc = ski.draw.polygon(x_arr, y_arr, image_dim)
                masked_arr[rr,cc] = 1

            masked_arr = masked_arr.astype(np.uint8)
            np.save(os.path.join(args.output, image_id + '_mask.npy'), masked_arr)
            #ski.io.imsave(os.path.join(args.output, image_id + '_mask.jpg'), masked_image, check_contrast=False)
        except:
            print('Error with: ', image_id)
            exit()

def main():
    parser = create_parser()
    args = parser.parse_args()
    #if args.aoi in ('AOI_1', 'AOI_2', 'AOI_3', 'AOI_4', 'AOI_5'):
    parse_aoi(args)

if __name__ == "__main__":
    main()