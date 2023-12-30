# Multi-Level Alignment Network for Cross-Domain Ship Detection

### 1. Introduction

This is the reserch code of the Remote sensing paper “Multi-Level Alignment Network for Cross-Domain Ship Detection”.

[Xu C, Zheng X, Lu X. Multi-Level Alignment Network for Cross-Domain Ship Detection. Remote Sensing. 2022; 14(10):2389. https://doi.org/10.3390/rs14102389](https://doi.org/10.3390/rs14102389)

Ship detection is an important research topic in the field of remote sensing. Compared with optical detection methods, Synthetic Aperture Radar (SAR) ship detection can penetrate clouds to detect hidden ships in all-day and all-weather. Currently, the state-of-the-art methods exploit convolutional neural networks to train ship detectors, which require a considerable labeled dataset. However, it is difficult to label the SAR images because of expensive labor and well-trained experts. To address the above limitations, this paper explores a cross-domain ship detection task, which adapts the detector from labeled optical images to unlabeled SAR images. There is a significant visual difference between SAR images and optical images. To achieve cross-domain detection, the multi-level alignment network, which includes image-level, convolution-level, and instance-level, is proposed to reduce the large domain shift. First, image-level alignment exploits generative adversarial networks to generate SAR images from the optical images. Then, the generated SAR images and the real SAR images are used to train the detector. To further minimize domain distribution shift, the detector integrates convolution-level alignment and instance-level alignment. Convolution-level alignment trains the domain classifier on each activation of the convolutional features, which minimizes the domain distance to learn domain-invariant features. Instance-level alignment reduces domain distribution shift on the features extracted from the region proposals. The entire multi-level alignment network is trained end-to-end and its effectiveness is proved on multiple cross-domain ship detection datasets.

### 2. Start
  Train a model by：
  
            python train.py --dataset sysu


  - `--dataset`: select dataset between "sysu" or "regdb".

#### The structure of code is shown below:
1. **configs**  
&emsp;|- da_faster_rcnn
&emsp;&emsp;&ensp;&thinsp;|- e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml		&emsp;&emsp;&emsp;---The Configuration File of the Dataset DIOR->HRSID
&emsp;&emsp;&ensp;&thinsp;|- e2e_da_faster_rcnn_R_50_C4_hrrsd_to_ssdd.yaml		&emsp;&emsp;&emsp;---The Configuration File of the Dataset HRRSD->SSDD
&emsp;&emsp; …
&emsp;…

2. **maskrcnn_benchmark**		 	&emsp;&emsp;---The Main Implementation File
&ensp;&thinsp;|- data			&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;--- Dataset Reading Code
&ensp;&thinsp;|- modeling		&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;---Detector Code
&ensp;&thinsp;|- backbone
&ensp;&thinsp;|- cyclegan 		&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;---Image Layer Alignment
&ensp;&thinsp;|- da_heads 		&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;---Convolutional and Instance Layer Alignment
&ensp;&thinsp;|- detector 
&emsp; …
…

3. **tools**
&ensp;&thinsp;|- train_net.py
&ensp;&thinsp;|- test_net.py
&emsp;…
…

#### Operating Instructions
1. **training commands**: 
             
            CUDA_VISIBLE_DEVICES=_device_ python tools/train_net.py --config-file _configuration file_
            
   - `--_device_`: select the GPU device number, for example: 0.
   - `_configuration file_`: select the config file address,  for example: "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml"

2. **test commands**: 
             
            CUDA_VISIBLE_DEVICES=_device_ python tools/test_net.py --config-file _configuration file_ MODEL.WEIGHT _model for test_

            
   - `--_device_`: select the GPU device number, for example: 0.
   - `_configuration file_`: select the config file address,  for example: "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml".  
   - `_model for test_`: select the model weight file address,  for example: model/end2end/dior_hrsid_1/model_0120000.pth
            

#### Dataset

The datasets used in this implementation include DIOR, HRRSD, HRSID, HRSID_Inshore, and SSDD. 
   Use the python files in maskrcnn_benchmark/data/datasets/ to transform the datasets and put them in the **datasets** folder.
    
### 3. Related work 

If you find the code and dataset useful in your research, please consider citing:
 
    @article{Xu2022Multi-Level,
    title={Multi-Level Alignment Network for Cross-Domain Ship Detection},
    author={Chujie Xu and Xiangtao Zheng and Xiaoqiang Lu},
    journal={Remote sensing},
    volume={14},
    number={10},
    pages={2389},
    year={2022}
    }


