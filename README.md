# Boundary-aware context neural network for medical image segmentation [MedIA'22]

## Introduction

In this study, we formulate a boundary-aware context neural network (BA-Net) for 2D medical image segmentation to capture richer context and preserve fine spatial information, which incorporates encoder-decoder architecture. In each stage of the encoder sub-network, a proposed pyramid edge extraction module first obtains multi-granularity edge information. Then a newly designed mini multi-task learning module for jointly learning segments the object masks and detects lesion boundaries, in which a new interactive attention layer is introduced to bridge the two tasks. In this way, information complementarity between different tasks is achieved, which effectively leverages the boundary information to offer strong cues for better segmentation prediction. Finally, a cross feature fusion module acts to selectively aggregate multi-level features from the entire encoder sub-network. By cascading these three modules, richer context and fine-grain features of each stage are encoded.

![image](img/overview.png)

## Update

2022/5: the code released.

## Usage

1. Install pytorch 

   - The code is tested on python 3.7 and pytorch 1.2.0.

2. Dataset
   
   You can download original datasets:
   - ISIC-2017: https://medicalsegmentation.com/covid19/
   - 
   - Please put dataset in folder `./data/covid_19_seg/`

3. Train and test

   - please run the following code for a quick start:
   -
   ```
   kkk
   ```

## Reference

If you consider use this code, please cite our paper:

```

@article{wang2022boundary,
  title={Boundary-aware context neural network for medical image segmentation},
  author={Wang, Ruxin and Chen, Shuyuan and Ji, Chaojie and Fan, Jianping and Li, Ye},
  journal={Medical Image Analysis},
  volume={78},
  pages={102395},
  year={2022},
  publisher={Elsevier}
}

```

     
 
