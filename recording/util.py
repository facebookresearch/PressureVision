# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import pickle
import os
import json
from pathlib import Path
import torch
import threading
import argparse
import yaml
import glob
from types import SimpleNamespace


SENSEL_COUNTS_TO_NEWTON = 1736   # How many force counts are approximately equal to one gram
SENSEL_PIXEL_PITCH = 0.00125    # The size of each force pixel
SENSEL_MAX_VIS = 20

CONFIG_BASE_PATH = './config'


def scalar_to_classes(scalar, thresholds):
    """
    Bins a float scalar into integer class indices.
    :param scalar: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :return:
    """
    if torch.is_tensor(scalar):
        out = torch.zeros_like(scalar, dtype=torch.int64)
    else:
        out = np.zeros_like(scalar, dtype=np.int64)

    for idx, threshold in enumerate(thresholds):
        out[scalar >= threshold] = idx  # may overwrite the same value many times

    return out


def classes_to_scalar(classes, thresholds):
    """
    Converts an integer class array into floating values. Obviously some discretization loss here
    :param classes: any shape, pytorch or numpy
    :param thresholds: list of thresholds. must be ascending
    :param final_value: if greater than the last threshold, fill in with this value
    :return:
    """
    if torch.is_tensor(classes):    # fill with negative ones
        out = -torch.ones_like(classes, dtype=torch.float)
    else:
        out = -np.ones_like(classes, dtype=np.float)

    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            val = thresholds[0]
        elif idx == len(thresholds) - 1:
            final_value = thresholds[-1] + (thresholds[-1] - thresholds[-2]) / 2    # Set it equal to the last value, plus half to gap to the previous thresh
            val = final_value
        else:
            val = (thresholds[idx] + thresholds[idx + 1]) / 2

        out[classes == idx] = val

    if out.min() < 0:
        raise ValueError('Thresholds were not broad enough')

    return out


def argparse_get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    return load_config(args.config)


def load_config(config_name):
    config_path = os.path.join(CONFIG_BASE_PATH, config_name + '.yml')

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = config_name
    return data_obj


def find_latest_checkpoint(config):
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    """
    all_checkpoints = glob.glob('data/model/*.pt')
    possible_matches = []
    for p in all_checkpoints:
        f = os.path.basename(p)
        if not f.startswith(config.CONFIG_NAME):
            continue
        f = f[len(config.CONFIG_NAME):-4] # cut off the prefix and suffix
        if not f.lower().islower():     # if it has any letters
            possible_matches.append(p)

    if len(possible_matches) == 0:
        raise ValueError('No valid model checkpoint files found')

    latest_file = max(possible_matches, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    return latest_file


def resnet_preprocessor(rgb):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    rgb = rgb - mean
    rgb = rgb / std
    return rgb


def set_subframe(subframe_id, subframe, disp_frame, steps_x=2, steps_y=2, title=None, interp=cv2.INTER_LINEAR):
    """ Helper function when making a large image up of many tiled smaller images"""
    frame_x = disp_frame.shape[1]
    frame_y = disp_frame.shape[0]
    inc_x = frame_x // steps_x
    inc_y = frame_y // steps_y

    subframe = cv2.resize(subframe, (inc_x, inc_y), interpolation=interp)

    if title is not None:
        cv2.putText(subframe, str(title), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(subframe, str(title), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    start_x = int((subframe_id % steps_x) * inc_x)
    start_y = int(subframe_id // steps_x) * inc_y
    disp_frame[start_y:start_y + inc_y, start_x:start_x + inc_x] = subframe


def pressure_to_colormap(kPa, colormap=cv2.COLORMAP_INFERNO):
    # Rescale the force array and apply the colormap

    pressure_array = kPa * (255.0 / SENSEL_MAX_VIS)     # linear scaling
    pressure_array[pressure_array > 255] = 255

    force_color = cv2.applyColorMap(pressure_array.astype(np.uint8), colormap)
    return force_color


def convert_counts_to_newtons(input_array):
    return input_array / SENSEL_COUNTS_TO_NEWTON


def convert_counts_to_kPa(input_array):
    # convert to kilopascals
    force = convert_counts_to_newtons(input_array)
    pa = force / (SENSEL_PIXEL_PITCH ** 2)
    return pa / 1000


def convert_kPa_to_newtons(kPa):
    return kPa * 1000 * (SENSEL_PIXEL_PITCH ** 2)


def mkdir(path, cut_filename=False):
    if cut_filename:
        path = os.path.dirname(os.path.abspath(path))
    Path(path).mkdir(parents=True, exist_ok=True)


def pkl_read(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def pkl_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(os.path.dirname(path))

    with open(path, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def json_write(path, data, auto_mkdir=False):
    if auto_mkdir:
        mkdir(path, cut_filename=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_read(path):
    with open(path, 'rb') as f:
        return json.load(f)


class MovieWriter:
    def __init__(self, path, fps=30):
        self.writer = None
        self.path = path
        self.fps = fps

    def write_frame(self, frame):
        if self.writer is None:
            mkdir(self.path, cut_filename=True)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, (frame.shape[1], frame.shape[0]))
        self.writer.write(frame)

    def close(self):
        self.writer.release()


def to_cpu_numpy(obj):
    """Convert torch cuda tensors to cpu, numpy tensors"""
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_cpu_numpy(v)
            return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_cpu_numpy(v))
        return res
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise TypeError("Invalid type for move_to")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

