# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import argparse
import glob
import random
from recording.sequence_reader import SequenceReader


def vis_data(seq_path):
    print('Viewing sequence:', seq_path)
    seq_reader = SequenceReader(seq_path)

    t = 0
    while True:
        frame = seq_reader.get_overall_frame(t, overlay_force=True)

        cv2.imshow('frame', frame)
        t = (t + 1) % seq_reader.num_frames

        keycode = cv2.waitKey(30) & 0xFF
        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='')
    args = parser.parse_args()

    if len(args.path) == 0:
        vis_data(random.choice(glob.glob('data/test/*/*')))
    else:
        vis_data(args.path)

