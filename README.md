# Unsupervised-Spectral-Demosaicing
This is an official PyTorch implementation for "Unsupervised Spectral Demosaicing with Lightweight Spectral Attention Networks" (IEEE Transactions on Image Processing, vol. 33, pp. 1655-1669, 2024), authored by [Kai Feng†, Haijin Zeng†, Yongqiang Zhao, Seong G. Kong and Yuanyang Bu]. The article presents a comprehensive unsupervised spectral demosaicing (USD) framework based on the characteristics of spectral mosaic images. This framework encompasses a training method, model structure, transformation strategy, a well-fitted model selection strategy, and a lightweight spectral attention module. This article also presents Mosaic25, a real 
25-band hyperspectral mosaic image dataset featuring various objects, illuminations, and materials for benchmarking purposes.

The full article can be found [here](https://ieeexplore.ieee.org/abstract/document/10443845).

## Environment Setup

### Deep Learning Framework:
The code is implemented using TensorFlow 2.5.0. You can install it via pip:

```bash
pip install tensorflow==2.5.0
```

### Dataset Download:
The dataset used for training is available at [provide_link_to_dataset]. Please download and extract the dataset before proceeding.

## Training

### Training File:
The training script is located at `train.py`. You can start training by running this script:

```bash
python train.py
```

### Modifying Variables:
Before training, please make sure to modify the following variables in `train.py`:

- `dataset_path`: Path to the directory containing the dataset.
- `batch_size`: Batch size for training.
- `num_epochs`: Number of epochs for training.

## Evaluation

### Evaluation File:
The evaluation script is located at `evaluate.py`. You can evaluate the trained model using this script:

```bash
python evaluate.py
```

### Modifying Variables:
Before evaluation, ensure to modify the following variables in `evaluate.py`:

- `dataset_path`: Path to the directory containing the dataset.
- `model_path`: Path to the trained model weights.

## Citation
If you find this code useful, feel free to cite our work using the following BibTeX entry:

```
@article{your_article,
  title={Your Title},
  author={Your Name},
  journal={Journal Name},
  year={Year},
  volume={Volume},
  number={Number},
  pages={Page},
  doi={DOI}
}
```
