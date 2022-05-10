# Boundary-aware context neural network for medical image segmentation [MedIA'22]

## Introduction

The coronavirus disease 2019 (COVID-19) pandemic is spreading worldwide. Considering the limited clinicians and resources, and the evidence that computed tomography (CT) analysis can achieve comparable sensitivity, specificity and accuracy with reverse-transcription polymerase chain reaction, the automatic segmentation of lung infection from CT scans supplies a rapid and effective strategy for COVID-19 diagnosis, treatment and follow-up. It is challenging because the infection appearance has high intra-class variation and inter-class indistinction in CT slices. Therefore, a new context-aware neural network is proposed for lung infection segmentation. Specifically, the Autofocus and Panorama modules are designed for extracting fine details and semantic knowledge and capturing the long-range dependencies of the context from both peer-level and cross-level. Also, a novel structure consistency rectification is proposed for calibration by depicting the structural relationship between foreground and background.

![image](img/overview.png)

## Update

2021/8: the code released.

## Usage

1. Install pytorch 

   - The code is tested on python 3.7 and pytorch 1.2.0.

2. Dataset
   
   You can download original datasets
   - Download the [Covid-19](https://medicalsegmentation.com/covid19/) dataset.
   - Please put dataset in folder `./data/covid_19_seg/`

3. Train and test

   - please run the following code for a quick start:
   -
   '''
   kkk
   '''

## Reference

If you consider use this code, please cite our paper:

'''
@article{wang2022boundary,
  title={Boundary-aware context neural network for medical image segmentation},
  author={Wang, Ruxin and Chen, Shuyuan and Ji, Chaojie and Fan, Jianping and Li, Ye},
  journal={Medical Image Analysis},
  volume={78},
  pages={102395},
  year={2022},
  publisher={Elsevier}
}
'''
     
 
