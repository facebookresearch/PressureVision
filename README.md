# PressureVision: Estimating Hand Pressure from a Single RGB Image



This repository contains the code, models, and data used in the PressureVision paper. By observing small changes in the appearance of the hand such as tissue deformation, blood flow, pose changes, and cast shadows, PressureVision estimates the pressure a hand exerts on a surface from a single RGB image.

[[Paper and Supplementary]](https://arxiv.org/abs/2203.10385) [[ECCV Oral Video]](https://youtu.be/nUxHy43AlsQ) [[Unfiltered Results Video]](https://youtu.be/AiI3b5CSrbs)

A single RGB image (top) is the only input to PressureVisionNet, which estimates a pressure map (bottom left). Ideally, this pressure matches the ground truth as measured by a pressure sensor (bottom right).

![PressureVision](docs/results_1.gif)
![PressureVision](docs/results_2.gif)

## Installation

Changelog: 02-2023: Cleaned and updated the codebase to make it compatible with new versions of libraries. Released a smaller version of the dataset and increased compatibility of the code. 

PressureVision requires Python and PyTorch, and has been tested with Python 3.10 and PyTorch 1.12 on Ubuntu. The following commands can be used to create a fresh conda environment and install all dependencies, however installing PyTorch may require tweaking:

```
# Clone repository and create new conda environment
git clone https://github.com/facebookresearch/PressureVision.git
cd PressureVision
conda create -n PressureVision python=3.10
conda activate PressureVision

# Install dependencies
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Downloading the Models and PressureVisionDB Dataset

PressureVisionDB consists of 36 participants recorded by 4 cameras for 16 hours. The participants are given fake names to facilitate readability of the dataset. The full dataset used in the paper was sampled at 15 FPS, and was 960 GB in total. As this is difficult to work with, we have released a small 140 GB dataset sampled at 3 FPS with a lower JPEG quality. Models trained on the small dataset perform within 1% volumetric IoU as with the full dataset.

The model weights and small dataset can be downloaded with the following command. Please use the `--full` flag if you want to download the full dataset.
```
python -m recording.downloader
```

## Getting Started

To run PressureVisionNet on the test set and generate a video of the results, run the following command. The video is saved to `data/movies`
```
python -m prediction.make_network_movie --config paper
```

To run a demo of PressureVisionNet on a realtime webcam, run the following command. Note that since the model was trained on a relatively fixed environment, some effort is needed to get it to generalize to your setup. We recommend mounting a webcam about 60 cm above a white table, and pointing it at a 45 degree angle downwards, as pictured below:
```
python -m prediction.webcam_demo --config paper
```



## Training and Evaluating

To train PressureVisionNet using the same hyperparameters as used in the paper, run:

```
python -m prediction.trainer --config paper
```

To generate the metrics reported in the paper, run the following command:

```
python -m prediction.evaluator --config paper --eval_on_test_set
```

## Other helpful commands

To visualize the raw data from a random sequence in the PressureVisionDB dataset, run:
```
python -m recording.view_recording
```

## License

The dataset and code for this project are released under the MIT License