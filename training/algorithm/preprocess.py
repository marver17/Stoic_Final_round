import numpy as np
import SimpleITK as sitk
from algorithm.lungmask import mask

def crop(mask,image):
    mask[mask == 2] = 1
    label = sitk.GetImageFromArray(mask,sitk.sitkInt8)
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(label)
    x1,y1,z1,x2,y2,z2 = lsif.GetBoundingBox(1)
    L = np.array([x1,y1,z1] , dtype='int').tolist()
    x,y,z = label.GetSize()
    U =[x - x1 - x2,y - y1- y2,z - z1 - z2]
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(L)
    crop.SetUpperBoundaryCropSize(U)
    label_crop  = crop.Execute(label)
    image_crop = crop.Execute(image)
    for k in image.GetMetaDataKeys():
        image_crop.SetMetaData(k,image.GetMetaData(k))
    return label_crop,image_crop





def preprocess(input_image: sitk.Image) -> sitk.Image:
    label = mask.apply(input_image)
    label_crop,image_crop = crop(label,input_image)
    return label_crop,image_crop



