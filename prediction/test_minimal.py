# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import DataLoader
import recording.util as util
from prediction.loader import visualize_batch
from prediction.model_builder import build_model


def test():
    model_dict = build_model(config, device, ['val'])

    best_model = torch.load(util.find_latest_checkpoint(config))
    best_model.eval()

    test_dataloader = DataLoader(model_dict['val_dataset'], batch_size=1, shuffle=True, num_workers=0)

    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            image = batch['img']
            force_gt = batch['force']
            predicted_force = best_model(image.cuda())

            if config.FORCE_CLASSIFICATION:
                predicted_force = torch.argmax(predicted_force, dim=1).detach()

            predicted_force = util.to_cpu_numpy(predicted_force)

        visualize_batch(
            image=util.to_cpu_numpy(batch['img_original']),
            pressure_gt=force_gt,
            pressure_est=predicted_force
        )


if __name__ == "__main__":
    config = util.parse_config_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test()
