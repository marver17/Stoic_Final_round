#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: admin_marioverd
"""
from monai.apps import load_from_mmar
from monai.data import decollate_batch
from algorithm.inferenza import sliding_window_inference
from monai.transforms import Activationsd,ScaleIntensityRanged,EnsureChannelFirstd, AsDiscrete,ToTensord, Compose,CropForegroundd, LoadImaged, SaveImage, Spacingd, Invertd
from monai.data import Dataset
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import sys, os

def list_split(listA, n):
    for x in range(0, len(listA), n):
        every_chunk = listA[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [None for y in range(n-len(every_chunk))]
        yield every_chunk


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def lesion_segmentation(path_image,output_dir):
    blockPrint()
    if path_image != None : 
        
        val_files = [
            {"img": path_image}
            ]
        val_transforms = Compose(
            [
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
                Spacingd(keys=["img"],pixdim=(0.8,0.8,5),mode='bilinear',align_corners= True),
                ScaleIntensityRanged(keys = ['img'], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["img"],source_key="img"),
                ToTensord(keys = ['img'])
            ])
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=4
        )

        post_trans = Compose(AsDiscrete(argmax=True),
                             Activationsd(keys=['val_outputs'], softmax=True),
                             Invertd(
                                 keys="pred",  # invert the `pred` data field, also support multiple fields
                                 transform=val_transforms,
                                 orig_keys="img",
                                  meta_keys="pred_meta_dict",
                                  orig_meta_keys="img_meta_dict",
                                  meta_key_postfix="meta_dict",
                                  nearest_interp=False,
                              ))
        model = load_from_mmar(
            "clara_pt_covid19_ct_lesion_segmentation_1", mmar_dir="./algorithm",
            pretrained=True).to(device)
        saver = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="seg", separate_folder=False)
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["img"].to(device)
                roi_size = (112, 112, 32)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size,model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                meta_data = decollate_batch(val_data["img_meta_dict"])
                for val_output, data in zip(val_outputs, meta_data):
                      saver(val_output, data)
    else :
        pass            

def P_lesion_segmentation(n_processes,path_img,output_dir):
        blockPrint()

        path_image_splitted = list(list_split(path_img,n_processes))
        processes = []
        for path_image in tqdm(path_image_splitted):
                i=0
                blockPrint()
                for rank in range(n_processes):
                    p = mp.Process(target=lesion_segmentation,
                                   args=(path_image[i], output_dir))
                    p.start()
                    processes.append(p)
                    i += 1
                for p in processes:
                    p.join()
