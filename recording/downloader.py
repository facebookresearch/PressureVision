# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import requests
import argparse
import zipfile
from tqdm import tqdm
from recording.util import mkdir

download_zips = {
    'test/9010_cray_crane.zip':                 'https://www.dropbox.com/s/mqmlypbhh17cxmv/9010_cray_crane.zip?dl=1',
    'test/9011_pug_rahl.zip':                   'https://www.dropbox.com/s/zkki9adz4xj06nh/9011_pug_rahl.zip?dl=1',
    'test/9014_jean_anthony.zip':               'https://www.dropbox.com/s/r5ypnry8jwzrqiu/9014_jean_anthony.zip?dl=1',
    'test/9019_eden_lawrence.zip':              'https://www.dropbox.com/s/wo5ezuym4hqz7hg/9019_eden_lawrence.zip?dl=1',
    'test/9025_patricia_petty.zip':             'https://www.dropbox.com/s/80wrfg6xktgfzrs/9025_patricia_petty.zip?dl=1',
    'test/9036_dellie_roy.zip':                 'https://www.dropbox.com/s/3rskhp0cblsiqn4/9036_dellie_roy.zip?dl=1',
    'train_fold_1/9002_gwydion_black.zip':      'https://www.dropbox.com/s/ant48d0d81fps1g/9002_gwydion_black.zip?dl=1',
    'train_fold_1/9009_murlynd_rune.zip':       'https://www.dropbox.com/s/an672rw160gdrac/9009_murlynd_rune.zip?dl=1',
    'train_fold_1/9017_charles_gibson.zip':     'https://www.dropbox.com/s/m9uvuorec14aoa0/9017_charles_gibson.zip?dl=1',
    'train_fold_1/9023_alec_ben.zip':           'https://www.dropbox.com/s/euqtxf3yhl24spa/9023_alec_ben.zip?dl=1',
    'train_fold_1/9029_cassie_fitzgerald.zip':  'https://www.dropbox.com/s/28tj4j3n11uo59w/9029_cassie_fitzgerald.zip?dl=1',
    'train_fold_1/9034_evard_dresden.zip':      'https://www.dropbox.com/s/iq3t74vtnel4owt/9034_evard_dresden.zip?dl=1',
    'train_fold_2/9003_julian_case.zip':        'https://www.dropbox.com/s/4tb5j878y7ktoet/9003_julian_case.zip?dl=1',
    'train_fold_2/9012_archie_mccullough.zip':  'https://www.dropbox.com/s/h6hsggac1fvpjs9/9012_archie_mccullough.zip?dl=1',
    'train_fold_2/9018_winifer_dee.zip':        'https://www.dropbox.com/s/oqfbhk0pbds7ap5/9018_winifer_dee.zip?dl=1',
    'train_fold_2/9024_leanardo_mcdaniel.zip':  'https://www.dropbox.com/s/o0xa40qk6d1da1x/9024_leanardo_mcdaniel.zip?dl=1',
    'train_fold_2/9030_haley_rosmerta.zip':     'https://www.dropbox.com/s/lhsh69k7pjuw7gy/9030_haley_rosmerta.zip?dl=1',
    'train_fold_2/9035_griselda_hollow.zip':    'https://www.dropbox.com/s/9o7z5m4dlfl433y/9035_griselda_hollow.zip?dl=1',
    'train_fold_3/9006_jane_fox.zip':           'https://www.dropbox.com/s/qwvn8r7yv38czq3/9006_jane_fox.zip?dl=1',
    'train_fold_3/9013_john_archer.zip':        'https://www.dropbox.com/s/wu1hnsggu298t15/9013_john_archer.zip?dl=1',
    'train_fold_3/9020_nystul_baxter.zip':      'https://www.dropbox.com/s/3dyheg2qja3fsnx/9020_nystul_baxter.zip?dl=1',
    'train_fold_3/9026_beth_kane.zip':          'https://www.dropbox.com/s/ld6fc8gg2dgd80x/9026_beth_kane.zip?dl=1',
    'train_fold_3/9031_rhys_gibbs.zip':         'https://www.dropbox.com/s/m1owakjedxj6udh/9031_rhys_gibbs.zip?dl=1',
    'train_fold_3/9037_jasmine_meyers.zip':     'https://www.dropbox.com/s/1wryf03fwp45tku/9037_jasmine_meyers.zip?dl=1',
    'train_fold_4/9007_mustrum_buchanan.zip':   'https://www.dropbox.com/s/z55rkrl19e25yqu/9007_mustrum_buchanan.zip?dl=1',
    'train_fold_4/9015_kovertol_potter.zip':    'https://www.dropbox.com/s/jy99rd1d1eua8b1/9015_kovertol_potter.zip?dl=1',
    'train_fold_4/9021_owen_osullivan.zip':     'https://www.dropbox.com/s/pkrsszh6lo0xc7f/9021_owen_osullivan.zip?dl=1',
    'train_fold_4/9027_mason_beck.zip':         'https://www.dropbox.com/s/lsl9mniugmlcc90/9027_mason_beck.zip?dl=1',
    'train_fold_4/9032_emily_wallace.zip':      'https://www.dropbox.com/s/wg5x4sjx86om6ba/9032_emily_wallace.zip?dl=1',
    'train_fold_4/9038_lottie_shaw.zip':        'https://www.dropbox.com/s/ii6eashs3wj82t9/9038_lottie_shaw.zip?dl=1',
    'val_fold_5/9008_beryl_cabric.zip':         'https://www.dropbox.com/s/3asuynuo9uf5swm/9008_beryl_cabric.zip?dl=1',
    'val_fold_5/9016_frankie_brady.zip':        'https://www.dropbox.com/s/gf1r3cx0wyffu1c/9016_frankie_brady.zip?dl=1',
    'val_fold_5/9022_herbert_stokes.zip':       'https://www.dropbox.com/s/hdpi9alko5v8amx/9022_herbert_stokes.zip?dl=1',
    'val_fold_5/9028_bailey_larson.zip':        'https://www.dropbox.com/s/qlavuo6372ktu99/9028_bailey_larson.zip?dl=1',
    'val_fold_5/9033_rowan_burch.zip':          'https://www.dropbox.com/s/ugkicn91uw8ekq0/9033_rowan_burch.zip?dl=1',
    'val_fold_5/9039_tiana_gordon.zip':         'https://www.dropbox.com/s/nm52v65xccd8e31/9039_tiana_gordon.zip?dl=1',
}

SAVE_ROOT_DIR = 'data'


def download_model_checkpoint():
    save_path = os.path.join(SAVE_ROOT_DIR, 'model/paper_59.pth')
    download_file_from_url('https://www.dropbox.com/s/abyhtojj972rzh8/paper_59.pth?dl=1', save_path)


def download_all_zips():
    for file_path in download_zips:
        save_zip_path = os.path.join(SAVE_ROOT_DIR, file_path)
        final_dir_path = os.path.splitext(save_zip_path)[0]     # Strip the extension

        download_file_from_url(download_zips[file_path], save_zip_path)     # Download the file

        print('Unzipping:', save_zip_path)
        with zipfile.ZipFile(save_zip_path, 'r') as zip_ref:
            zip_ref.extractall(final_dir_path)

        os.remove(save_zip_path)    # Delete the zip file


def download_file_from_url(url, filename):
    """
    Download file from a URL to filename,
    displaying progress bar with tqdm
    taken from https://stackoverflow.com/a/37573701
    and https://github.com/facebookresearch/ContactPose/blob/main/utilities/networking.py
    """

    print('Downloading:', filename)
    mkdir(filename, cut_filename=True)

    try:
        r = requests.get(url, stream=True)
    except ConnectionError as err:
        print(err)
        return False

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    done = True
    datalen = 0
    with open(filename, 'wb') as f:
        itr = r.iter_content(block_size)
        while True:
            try:
                try:
                    data = next(itr)
                except StopIteration:
                    break

                t.update(len(data))
                datalen += len(data)
                f.write(data)
            except KeyboardInterrupt:
                done = False
                print('Cancelled')
            except ConnectionError as err:
                done = False
                print(err)

    t.close()

    if (not done) or (total_size != 0 and datalen != total_size):
        print("ERROR, something went wrong")
        try:
            os.remove(filename)
        except OSError as e:
            print(e)
        return False
    else:
        return True


if __name__ == "__main__":
    download_model_checkpoint()
    download_all_zips()

