# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import recording.util as util
from prediction.model_builder import build_model
import prediction.evaluator as evaluator


def val_epoch(config, val_metrics):
    model.eval()
    loss_meter = util.AverageMeter('Loss', ':.4e')
    evaluator.reset_metrics(val_metrics)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader)):
            image = data['img']
            force_gt = data['force']
            image_gpu = image.cuda()
            force_gt_gpu = force_gt.cuda()
            batch_size = image.shape[0]

            force_estimated = model(image_gpu)
            loss = criterion(force_estimated, force_gt_gpu)

            loss_meter.update(loss.item(), batch_size)

            if config.FORCE_CLASSIFICATION:
                force_pred_class = torch.argmax(force_estimated, dim=1)
                force_pred_scalar = util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
            else:
                force_pred_scalar = force_estimated.squeeze(1) * config.NORM_FORCE_REGRESS

            force_gt_scalar = data['raw_force'].cuda()
            evaluator.run_metrics(val_metrics, force_gt_scalar, force_pred_scalar, config)

    writer.add_scalar('val/loss', loss_meter.avg, global_iter)
    for key, metric in val_metrics.items():
        writer.add_scalar('val/' + key, metric.compute(), global_iter)

    writer.flush()
    print('Finished val epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_meter.avg))


def train_epoch(config):
    model.train()
    loss_meter = util.AverageMeter('Loss', ':.4e')

    iterations = 0
    global global_iter

    with tqdm(total=config.TRAIN_ITERS_PER_EPOCH) as progress_bar:
        for idx, data in enumerate(train_loader):
            image = data['img']
            force_gt = data['force']
            image_gpu = image.cuda()
            force_gt_gpu = force_gt.cuda()
            batch_size = image.shape[0]

            force_estimated = model(image_gpu)
            loss = criterion(force_estimated, force_gt_gpu)

            loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += batch_size
            global_iter += batch_size
            progress_bar.update(batch_size)     # Incremental update
            progress_bar.set_postfix(loss=str(loss_meter))
            if iterations >= config.TRAIN_ITERS_PER_EPOCH:
                break

    writer.add_scalar('training/loss', loss_meter.avg, global_iter)
    writer.add_scalar('training/lr', scheduler.get_last_lr()[0], global_iter)
    print('Finished training epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_meter.avg))
    writer.flush()


if __name__ == "__main__":
    config = util.argparse_get_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = build_model(config, device, ['train', 'val'])
    criterion = model_dict['criterion']
    model = model_dict['model']

    train_loader = DataLoader(model_dict['train_dataset'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(model_dict['val_dataset'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config.LEARNING_RATE_INITIAL)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LEARNING_RATE_SCHEDULER_STEP,
                                                     gamma=config.LEARNING_RATE_SCHEDULER_GAMMA)

    desc = config.CONFIG_NAME + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter('runs/' + desc)
    global_iter = 0

    val_metrics = evaluator.setup_metrics(device)

    for epoch in range(config.MAX_EPOCHS):
        train_epoch(config)
        val_epoch(config, val_metrics)
        torch.save(model.state_dict(), 'data/model/{}_{}.pt'.format(config.CONFIG_NAME, epoch))
        scheduler.step()
        print('\n')

    evaluator.evaluate(config, device, force_eval_on_test_set=True)
