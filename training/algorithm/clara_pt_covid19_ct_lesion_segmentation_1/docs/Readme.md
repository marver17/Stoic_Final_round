# Disclaimer
This training and inference pipeline was developed by NVIDIA. It is based on a segmentation and classification model developed by NVIDIA researchers in conjunction with the NIH.

# Model Overview
A pre-trained model for volumetric (3D) segmentation of the COVID-19 lesion from CT images.

## Workflow
The model is trained using a 3D SegResNet [1].

![Diagram showing the flow from model input, through the model architecture, and to model output](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lesion_segmentation_workflow.png)

## Data
This model was trained on a global dataset with a large experimental cohort collected from across the globe. The CT volumes of 919 independent subjects are provided by NIH with experts’ lesion region annotations.

- Target: Lesion
- Task: Segmentation
- Modality: CT
- Size: 919 3D volumes (736 Training, 90 Validation, 93 Testing)
- Challenge: Large ranging foreground size

# Training configuration
The training was performed with the following:

- Script: train_multi_gpu.sh
- GPU: four (at least) 16GB of GPU memory
- Actual Model Input: 224 x 224 x 32
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-3
- Loss: DiceCELoss

**If out-of-memory or program crash occurs while caching the data set, please change the ``cache_rate`` in ``CacheDataset`` to a lower value in the range (0, 1).**

## Input
Input: 1 channel CT image with intensity in HU and arbitary spacing

1. Resampling spacing to (0.8, 0.8, 5) mm;
2. Clipping intensity to [-1000, 500] HU;
3. Converting to channel first;
4. Randomly cropping the volume to a fixed size (224, 224, 32);
5. Randomly applying spatial flipping;
6. Randomly applying spatial rotation;
6. Randomly shifting intensity of the volume.

## Output
Output: 2 channels
- Label 0: background
- Label 1: lesion

# Model Performance
Dice score is used for evaluating the performance of the model. On the test set, the trained model achieved score of 0.7109 for lesion.

## Training Performance
Training loss over 2000 epochs.

![Graph that shows training acc over 2000 epochs](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lesion_segmentation_train.png)

## Validation Performance
Validation mean dice score over 2000 epochs.

![Graph that shows validation mean dice getting higher over 2000 epochs until converging around 0.72](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_covid19_ct_lesion_segmentation_val.png)

# Intended Use
The model needs to be used with NVIDIA hardware and software. For hardware, the model can run on any NVIDIA GPU with memory greater than 16 GB. For software, this model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.  Find out more about Clara Train at the [Clara Train Collections on NGC](https://ngc.nvidia.com/catalog/collections/nvidia:claratrainframework).

**The Clara pre-trained models are for developmental purposes only and cannot be used directly for clinical procedures.**

# License
[End User License Agreement](https://developer.nvidia.com/clara-train-eula) is included with the product. Licenses are also available along with the model application zip file. By pulling and using the Clara Train SDK container and downloading models, you accept the terms and conditions of these licenses.

# References
[1] Myronenko, A., 2018, September. 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320). Springer, Cham. https://arxiv.org/pdf/1810.11654.pdf
