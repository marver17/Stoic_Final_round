#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:25:45 2022

@author: admin_marioverd
"""
from ctdataset import CTDataset
import SimpleITK as sitk
import numpy as np
import re


def age_sex_info(itk_image: sitk.Image) -> int :
    age = itk_image.GetMetaData('PatientAge')
    if any(c.isalpha() for c in age)  == True :
        age = int(re.findall('\d+', age )[0])
    else:
        age = int(age)

    sex = itk_image.GetMetaData('PatientSex')
    if sex == "F" :
        sex = 0
    else:
        sex = 1
    return age, sex

def componenti_connesse(sitk_image):
    img = sitk.Cast(sitk_image,sitk.sitkInt8)
    con = sitk.ConnectedComponent(img)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(con)
    com_connesse = label_shape_filter.GetNumberOfLabels()
    return com_connesse         

def give_image(self,idx):
        sitk_image = self.read_image(idx)
        age,sex = age_sex_info(sitk_image)
        vox_dim = np.prod(np.array(sitk_image.GetSpacing()))
        return sitk.GetArrayFromImage(sitk_image),age,sex,vox_dim


def extract_all_(path_image):
        image,age,sex,vox_dim = give_image(path_image)
    
    
        
        # age,sex = self.get_age_sex(idx)
        lung_mask = self.read_lung_mask(idx)    
        lesion_mask = self.read_lesion_mask(idx)
        vol_lesion = np.sum(np.bool_(lesion_mask))
        n_con  = self.componenti_connesse(idx)
        if vol_lesion != 0:
            vol_masklung = np.sum(np.bool_(lung_mask))
            vol_lesion = np.sum(np.bool_(lesion_mask))/vol_masklung
            img = image[lesion_mask == 1]
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            check_lesion = 1
            return ID,age,sex,vol_lesion,mean_intensity,std_intensity,check_lesion,prob_COVID,prob_Severe,n_con
        else :
            return ID,age,sex,0,0,0,0,prob_COVID,prob_Severe,0


class features_exctractor(CTDataset):
    
    def __init__(self, data):
        self.data = data
        
    # def get_age_sex(self,idx) :
    #     data = self.data
    #     sitk_image = sitk.ReadImage(data.get_x_filename(idx))
    #     age,sex = age_sex_info(sitk_image)
    #     return age,sex        
    
    def read_image(self,idx):
        data = self.data
        sitk_image = sitk.ReadImage(data.get_x_filename_preprocessed(idx))
        return sitk_image
    
    
    def read_lesion_mask(self,idx):
        data = self.data
        sitk_image = sitk.ReadImage(data.get_x_filename_lessionmask(idx))
        return sitk.GetArrayFromImage(sitk_image)
    
    def read_lung_mask(self,idx):
        data = self.data
        sitk_image = sitk.ReadImage(data.get_x_filename_lungmask(idx))
        return sitk.GetArrayFromImage(sitk_image)
    
    
    def _all_(self,idx):
        image,age,sex,vox_dim,ID = self.give_image(idx)
        y = self.data.get_y(idx)
        prob_COVID,prob_Severe = y[0],y[1]
        # age,sex = self.get_age_sex(idx)
        lung_mask = self.read_lung_mask(idx)    
        lesion_mask = self.read_lesion_mask(idx)
        vol_lesion = np.sum(np.bool_(lesion_mask))
        n_con  = self.componenti_connesse(idx)
        if vol_lesion != 0:
            vol_masklung = np.sum(np.bool_(lung_mask))
            vol_lesion = np.sum(np.bool_(lesion_mask))/vol_masklung
            img = image[lesion_mask == 1]
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            check_lesion = 1
            return ID,age,sex,vol_lesion,mean_intensity,std_intensity,check_lesion,prob_COVID,prob_Severe,n_con
        else :
            return ID,age,sex,0,0,0,0,prob_COVID,prob_Severe,0


        
        
        
