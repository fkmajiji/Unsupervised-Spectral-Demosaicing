# Unsupervised-Spectral-Demosaicing
This is an official PyTorch implementation for "Unsupervised Spectral Demosaicing with Lightweight Spectral Attention Networks" (IEEE Transactions on Image Processing), authored by [Kai Feng†, Haijin Zeng†, Yongqiang Zhao, Seong G. Kong and Yuanyang Bu]. The article presents a comprehensive unsupervised spectral demosaicing (USD) framework based on the characteristics of spectral mosaic images. This framework encompasses a training method, model structure, transformation strategy, a well-fitted model selection strategy, and a lightweight spectral attention module. This article also presents Mosaic25, a real 
25-band hyperspectral mosaic image dataset featuring various objects, illuminations, and materials for benchmarking purposes.

The schematic diagram of the proposed unsupervised spectral demosaicing (USD) framework is below:
![image](https://github.com/fkmajiji/Unsupervised-Spectral-Demosaicing/assets/35318585/2b477726-2209-4e1d-9a5d-1bc79b5a9066)

The full article can be found [here](https://ieeexplore.ieee.org/abstract/document/10443845).

## Environment Setup

### Deep Learning Framework:
The code is implemented using Pytorch 1.7.0. 

### Dataset Download:

#### Synthetic dataset
There are two synthetic datasets used in the paper: ICVL and NTIRE.

The 25-band ICVL dataset is spectrally sampled from the original dataset using the spectral response functions provided by the IMEC manufacturer, which  is available at [here](https://drive.google.com/drive/folders/1zTyM5pDkrbMa6c7QjNiBfaCGLnfkRkYc?usp=sharing). Please download and extract the dataset before proceeding. The detailed sampling function is available at [here](https://github.com/fkmajiji/Spectral-Image-Sampling)

The 16-band NTIRE dataset is provided by NTIRE’s Spectral Demosaicing Challenge， which is available at [here](https://drive.google.com/file/d/1ZHAsmrGgF1P_GbNib4OZEleBhVPpmn5s/view) and [here](https://drive.google.com/file/d/16jltk-q3VTEfCEWIwy4z6KNBy0daUoQ_/view)

#### Real-world dataset: Mosaic25
We released one real-world 25-band mosaic image dataset for benchmarking purposes, which contains various 
objects, illuminations, and materials. It is available at [here](https://drive.google.com/drive/folders/1v1eWW0GJqxw734JJEvxneYDgOpch9Lm4?usp=sharing). There are two folders in the test folder. The ‘paper used’ is the 17 images used in the experimental part of the article, and the ‘supplemental’ folder is the images we added later. 

![image](https://github.com/fkmajiji/Unsupervised-Spectral-Demosaicing/assets/35318585/36bac75d-f702-4868-88b4-4bb93b3a4682)

## Training

### Training File:
The training script is located at `main_USD_opttrain_optsetup.py`. 

### Modifying Variables:
Before training, please make sure to modify the following variables:

- `dataset_path`: the input paths of functions 'get_training_set_opt', 'NTIREDatasetFromFolder', and 'get_real_mosaic_training_set_opt'. They are the paths to the directory containing the dataset.

## Evaluation

### Evaluation File:
The evaluation scripts for synthetic datasets with GT and real-world dataset without GT are located at `eval_iter_opt_synthetic_dataset.py` and `eval_iter_opt_real_dataset.py`. After running, there will be one CSV file containing the PSNR or SEI values.

### Self-evaluation index (SEI):
These two py files contain the self-supervised evaluation index (SEI) we proposed. SEI quantifies periodic distortion during 
overfitting caused by USD training. With this index, we can automatically select the well-fitted model from a multitude of checkpoints.

### Modifying Variables:
Before evaluation, ensure to modify the following variables:

- `type_name_list`: Folder name saved in the checkpoint folder during your training process.
- `for epoch_num in range(X, X, X)`: Please modify the scope of the for loop to evaluate a single model or a series of models. Please match the saved starting point and interval.

## Citation
If you find this code useful, feel free to cite our work using the following texts:

```
K. Feng, H. Zeng, Y. Zhao, S. G. Kong and Y. Bu, "Unsupervised Spectral Demosaicing With Lightweight Spectral Attention Networks," in IEEE Transactions on Image Processing, vol. 33, pp. 1655-1669, 2024, doi: 10.1109/TIP.2024.3364064.
```

```
@ARTICLE{10443845,
  author={Feng, Kai and Zeng, Haijin and Zhao, Yongqiang and Kong, Seong G. and Bu, Yuanyang},
  journal={IEEE Transactions on Image Processing}, 
  title={Unsupervised Spectral Demosaicing With Lightweight Spectral Attention Networks}, 
  year={2024},
  volume={33},
  number={},
  pages={1655-1669},
  keywords={Training;Correlation;Cameras;Task analysis;Hyperspectral imaging;Electronics packaging;Distortion;Spectral demosaicing;unsupervised learning;spectral imaging;spectral attention networks},
  doi={10.1109/TIP.2024.3364064}}
```

