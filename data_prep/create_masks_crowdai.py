#python3 /workspaces/BetaNet/data_prep/create_masks_crowdai.py -a /data/crowdai/train/annotation.json -i /data/crowdai/train/3band -o /data/crowdai/train/masks/ 

def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Utility to create masks.')
    parser.add_argument('-a', '--annotations',
                        metavar='/data/crowdAI/...',
                        dest='annotations',
                        #required=True,
                        help='specify full path for annotations JSON',
                        type=str)
    parser.add_argument('-i', '--img',
                        #required=True,
                        dest='img',
                        metavar='/data/crowdAI/...',
                        help='specify full path for images',
                        type=str)
    parser.add_argument('-o', '--output',
                        #required=True,
                        dest='output',
                        metavar='/data/crowdAI/...',
                        help='specify full path for output of mask image files',
                        type=str)
    
    return parser
def create_and_save_mask(output_path, coco, img_ann_dict, img_id):
    #global coco
    #global img_ann_dict

    import os
    import numpy as np
    img_info = coco.loadImgs(img_id)
    img_h = img_info[0]['height']
    img_w = img_info[0]['width']
    img_filename = img_info[0]['file_name']
    mask_filename = img_filename.split('.',1)[0]+'.npy'
    mask_path = os.path.join(output_path, mask_filename)
    img_arr_curr = np.zeros((img_w,img_h))
    for ann_id in img_ann_dict[img_id]:
        img_arr_curr += coco.annToMask(coco.loadAnns(ann_id)[0])
    
    np.save(mask_path, img_arr_curr)
    return

def parse_imgs(args): 
    import numpy as np
    from pathlib import Path
    import os
    import skimage.io as io
    from tqdm import tqdm
    from pycocotools.coco import COCO
    from pycocotools import mask as cocomask
    from joblib import Parallel, delayed

    coco = COCO(args.annotations)

    img_ids_all = coco.getImgIds()
    
    for i in range(0, len(img_ids_all), 10000):
        img_ids = img_ids_all[i:i+10000]

        print("Iteration: ", i)

        print("Gathering annotation IDs for each image ID...")
        img_ann_dict = {}
        for img_id in tqdm(img_ids):
            img_ann_dict[img_id] = coco.getAnnIds(img_id)

        print("Creating and writing masks...")
        for img_id in tqdm(img_ids):
            create_and_save_mask(args.output, coco, img_ann_dict, img_id)
    
def main():
    parser = create_parser()
    args = parser.parse_args()

    parse_imgs(args)

if __name__ == "__main__":
    main()