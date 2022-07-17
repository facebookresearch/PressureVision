# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
import torch.multiprocessing
import recording.util as util

disp_x = 480 * 2
disp_y = 384 * 2
aspect_ratio = 480 / 384


def run_model(img, best_model):
    # Takes in a cropped OpenCV-formatted image, does the preprocessing, runs the network, and the postprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    img = util.resnet_preprocessor(img)
    img = img.transpose(2, 0, 1).astype('float32')
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        force_pred_class = best_model(img.cuda())
        force_pred_class = torch.argmax(force_pred_class, dim=1)
        force_pred_scalar = util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        force_pred_scalar = force_pred_scalar.detach().squeeze().cpu().numpy()

    return force_pred_scalar


def crop_and_resize(img, scale):
    y_scale = int(scale / aspect_ratio)

    start_x_int = max(img.shape[1] // 2 - scale, 0)
    end_x_int = min(img.shape[1] // 2 + scale, img.shape[1])
    start_y_int = max(img.shape[0] // 2 - y_scale, 0)
    end_y_int = min(img.shape[0] // 2 + y_scale, img.shape[0])
    crop_frame = img[start_y_int:end_y_int, start_x_int:end_x_int, :]

    resize_frame = cv2.resize(crop_frame, (480, 384))

    return resize_frame


def run_demo():
    window_name = 'PressureVision'
    disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    scale = 355

    best_model = torch.load(util.find_latest_checkpoint(config))
    best_model.eval()

    def change_scale(value):
        nonlocal scale
        scale = value

    cv2.imshow(window_name, disp_frame)
    cv2.createTrackbar('scale', window_name, scale, 2000, change_scale)

    while True:
        ret, camera_frame = cap.read()
        if camera_frame is None:
            continue

        util.set_subframe(0, camera_frame, disp_frame, title='Raw Camera Frame')

        crop_frame = crop_and_resize(camera_frame, scale)
        # crop_frame = cv2.GaussianBlur(crop_frame, (0, 0), 1)   # Perform some blurring. This may help the network generalize slightly

        util.set_subframe(1, crop_frame, disp_frame, title='Network Input')

        force_pred = run_model(crop_frame, best_model)
        util.set_subframe(2, util.pressure_to_colormap(force_pred), disp_frame, title='Network Output')

        overlay_frame = cv2.addWeighted(crop_frame, 0.3, util.pressure_to_colormap(force_pred), 1.0, 0.0)
        util.set_subframe(3, overlay_frame, disp_frame, title='Network Output with Overlay')

        cv2.imshow(window_name, disp_frame)
        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    config = util.argparse_get_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_demo()
