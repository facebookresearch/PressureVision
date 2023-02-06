# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import os.path
import argparse
import torch
from tqdm import tqdm
import torchmetrics
import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
import recording.util as util
from prediction.model_builder import build_model


class VolumetricIOU(torchmetrics.Metric):
    """
    This calculates the IoU summed over the entire dataset, then averaged. This means an image with no
    GT or pred force will contribute none to this metric.
    """
    full_state_update = True

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape

        high = torch.maximum(preds, target)
        low = torch.minimum(preds, target)

        self.numerator += torch.sum(low)
        self.denominator += torch.sum(high)

    def compute(self):
        return self.numerator / self.denominator


class ContactIOU(torchmetrics.Metric):
    full_state_update = True

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Input of the form (batch_size, img_y, img_x)
        assert preds.shape == target.shape
        assert preds.dtype == torch.long    # Make sure we're getting ints

        bool_pred = preds > 0
        bool_gt = target > 0

        self.numerator += torch.sum(bool_gt & bool_pred)
        self.denominator += torch.sum(bool_gt | bool_pred)

    def compute(self):
        return self.numerator / self.denominator


def reset_metrics(all_metrics):
    for key, metric in all_metrics.items():
        metric.reset()


def print_metrics(all_metrics, network_name='', save=True):
    out_dict = dict()
    for key, metric in all_metrics.items():
        val = metric.compute().item()
        print(key, val)
        out_dict[key] = val

    if save:
        d = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_filename = os.path.join('data', 'eval', f"{os.path.basename(network_name)}_{d}.txt")
        util.json_write(out_filename, out_dict, auto_mkdir=True)


def run_metrics(all_metrics, pressure_gt, pressure_est, config):
    # Takes CUDA batched tensors as input

    contact_pred = (pressure_est > config.CONTACT_THRESH).long()
    contact_gt = (pressure_gt > config.CONTACT_THRESH).long()

    all_metrics['contact_iou'](contact_pred, contact_gt)

    all_metrics['mse'](pressure_est, pressure_gt)
    all_metrics['mae'](pressure_est, pressure_gt)
    all_metrics['vol_iou'](pressure_est, pressure_gt)

    any_contact_pred = torch.sum(contact_pred, dim=(1, 2)) > 0
    any_contact_gt = torch.sum(contact_gt, dim=(1, 2)) > 0

    all_metrics['temporal_accuracy'](any_contact_pred, any_contact_gt)


def setup_metrics(device):
    all_metrics = dict()

    all_metrics['contact_iou'] = ContactIOU().to(device)
    all_metrics['mse'] = torchmetrics.MeanSquaredError().to(device)
    all_metrics['mae'] = torchmetrics.MeanAbsoluteError().to(device)
    all_metrics['vol_iou'] = VolumetricIOU().to(device)
    all_metrics['temporal_accuracy'] = torchmetrics.Accuracy(task='binary').to(device)
    return all_metrics


def evaluate(config, device, force_eval_on_test_set=False):

    if force_eval_on_test_set:
        print('Testing on test set!')
        config.VAL_FILTER = ['data/test/*/*']
        config.DATALOADER_TEST_SKIP_FRAMES = 1
    else:
        config.DATALOADER_TEST_SKIP_FRAMES = 4  # Test on 1/4 of frames to speed up testing
        print('Testing on validation set!')

    random.seed(5)  # Set the seed so the sequences will be randomized the same
    model_dict = build_model(config, device, ['val'])

    checkpoint_path = util.find_latest_checkpoint(config)
    best_model = model_dict['model']
    best_model.load_state_dict(torch.load(checkpoint_path))
    best_model.eval()

    val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    all_metrics = setup_metrics(device)

    for idx, batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            image = batch['img']
            force_gt_scalar = batch['raw_force'].cuda()

            if config.FORCE_CLASSIFICATION:
                force_pred_class = best_model(image.cuda())
                force_pred_class = torch.argmax(force_pred_class, dim=1)
                force_pred_scalar = util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = best_model(image.cuda()).squeeze(1) * config.NORM_FORCE_REGRESS
                force_pred_scalar = F.relu(force_pred_scalar, inplace=True)

            run_metrics(all_metrics, force_gt_scalar, force_pred_scalar, config)

    if force_eval_on_test_set:
        checkpoint_path += '_TEST_SET'

    print_metrics(all_metrics, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--eval_on_test_set', action='store_true')  # Normally this script runs on the validation set. This flag forces using the test set
    parser.add_argument('-cfg', '--config', type=str, required=True)
    args = parser.parse_args()
    config = util.load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate(config, device, force_eval_on_test_set=args.eval_on_test_set)
