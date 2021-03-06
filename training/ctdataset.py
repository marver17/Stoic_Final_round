import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from algorithm.preprocess import preprocess
import torch.multiprocessing as mp
from algorithm.lesion_segmentation import list_split
from tqdm import tqdm
def process_for_all(path_image,name_cut,name_label):
    if path_image != None : 
            sitk_image = sitk.ReadImage(path_image)
            label,image = preprocess(sitk_image)
            sitk.WriteImage(label,name_label,False)
            sitk.WriteImage(image, name_cut, False)
            #print(f"Image {path_image} processed.")
    else :
        pass
    
class CTDataset(Dataset):
    def __init__(self, data, preprocess_dir):
        self.data = data
        self.preprocess_dir = os.path.join(preprocess_dir, "preprocess")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.data)

    def get_x_filename(self, idx):
        return self.data[idx]['x']

    def get_x_filename_preprocessed(self, idx):
        return os.path.join(self.preprocess_dir, os.path.basename(self.get_x_filename(idx)))

    def get_x_filename_lungmask(self, idx):
        return os.path.join(self.preprocess_dir, os.path.basename(self.get_x_filename(idx)[:-4] + "_lungmask.nii.gz"))

    def get_x_filename_lessionmask(self, idx):
        return os.path.join(self.preprocess_dir, os.path.basename(self.get_x_filename(idx)[:-4] + "_seg.nii.gz"))
    
    def is_preprocessed(self, idx):
        return os.path.isfile(self.get_x_filename_preprocessed(idx))

    def get_x_preprocessed(self, idx):
        sitk_image = sitk.ReadImage(self.get_x_filename_preprocessed(idx))
        return sitk.GetArrayFromImage(sitk_image)
    
    def list_is_no_preprocessed(self):
        a = [not(os.path.isfile(self.get_x_filename_preprocessed(idx))) for idx in range(self.__len__())]
        return a
    
    def list_is_no_lesionmask(self):
        a = [not(os.path.isfile(self.get_x_filename_lessionmask(idx))) for idx in range(self.__len__())]
        return a

    
    def get_x(self, idx):
        if not self.is_preprocessed(idx):
            sitk_image = sitk.ReadImage(self.get_x_filename(idx))
            label,image = preprocess(sitk_image)
            sitk.WriteImage(image, self.get_x_filename_preprocessed(idx), False)
            sitk.WriteImage(label,self.get_x_filename_lungmask(idx),False)
            #print(f"Image {idx} processed.")
    
    
    def get_y(self, idx):
        return np.asarray(self.data[idx]['y']).astype(np.float32)

    def __getitem__(self, idx):
        x = self.get_x(idx)
        x = self.transform(np.expand_dims(x, axis=0))
        x = torch.from_numpy(x)
        y = torch.from_numpy(self.get_y(idx))
        return x, y


    def get_path_list(self,num_imm,check_preprocess = False):
        
        if num_imm == "all":
            num_imm = self.__len__()
        else :
            pass
        a = [self.get_x_filename(x) for x in range(self.__len__())][:num_imm]
        b = [self.get_x_filename_preprocessed(x) for x in range(self.__len__())][:num_imm]
        c = [self.get_x_filename_lungmask(x) for x in range(self.__len__())][:num_imm]
        if check_preprocess == True:
            a = list(np.array(a)[self.list_is_no_preprocessed()[:num_imm]])
            b = list(np.array(b)[self.list_is_no_lesionmask()[:num_imm]])
            
        else :
            
                pass
        
        return a,b,c,num_imm
    

    def get_all_x(self,n_processes,num_imm,is_not_pre  = False):
        
                
                a,b,c,n_imm = self.get_path_list(num_imm,is_not_pre)
                path_image_splitted = list(list_split(a,n_processes))
                path_crop_splitted = list(list_split(b,n_processes))
                path_lung_splitted = list(list_split(c,n_processes))

                processes = []
                
                 
                for path_image,path_crop,path_lung in tqdm(zip(path_image_splitted,path_crop_splitted,path_lung_splitted),total =len(path_image_splitted)):
                        i=0
                        for rank in range(n_processes):
                            p = mp.Process(target=process_for_all,
                                           args=(path_image[i], path_crop[i],path_lung[i]))
                            p.start()
                            processes.append(p)
                            i += 1
                        for p in processes:

                            p.join()
                

# if __name__ == "__main__":
#     import sys
#     import glob
#     import matplotlib.pyplot as plt

#     preprocess_dir = sys.argv[1]
#     image_dir = sys.argv[2]

#     data = [
#         {'x': filename, 'y': [0, 0]}
#         for filename in glob.glob(os.path.join(image_dir, "*.mha"))
#     ]
#     ctdataset = CTDataset(data, preprocess_dir)

#     steps = 4
#     for x, y in ctdataset:
#         print(y)
#         print(x.shape)
#         x = x.numpy()[0]
#         length = x.shape[1]
#         start = length // 3
#         stop = (length // 5) * 4
#         step = (stop - start) // steps
#         fig, axes = plt.subplots(1, steps, figsize=(15, 4))
#         its = range(start, stop, step)
#         for it, axis in zip(its, axes):
#             screenshot = x[:, it, :][::-1]
#             axis.imshow(screenshot, cmap='gray')
#             axis.axis('off')
#         plt.suptitle(f'label: {y}')
#         plt.show()

