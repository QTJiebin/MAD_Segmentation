import os
import os.path as osp
import argparse
import numpy as np
from collections import Counter
from color_map import VOCColormap
from PIL import Image

COLOR_MAP = VOCColormap().get_color_map()

NameList = ['Background', 'Aero plane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse',
            'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'TvMonitor']


def count_labels(lbl):
    ct = Counter(lbl.reshape(-1))
    ct_TopN = ct.most_common()
    sorted_names = []
    for ct_top in ct_TopN:
        sorted_names.append(NameList[ct_top[0]])
    print(' > '.join(sorted_names),
          '\nNo other classes appear.' if len(sorted_names) < len(NameList) else '\n')


def cvt_npy2seg(npy_path, out_dir=None):
    lbl = np.load(npy_path).squeeze()
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        count_labels(lbl)
        lbl_pil = Image.fromarray(lbl.astype(np.uint8))
        lbl_pil.putpalette(COLOR_MAP)
        if out_dir:
            sav_path = osp.join(out_dir, osp.splitext(osp.basename(npy_path))[0] + '.png')
        else:
            sav_path = osp.splitext(npy_path)[0] + '.png'
        lbl_pil.save(sav_path)
        print('Segmentation saved:', sav_path)
    else:
        raise ValueError(
            '[%s] Cannot convert the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % npy_path
        )


'''
Run as below:

    python cvat2voc.py /labels/****.npy

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
    npy_file = f'{args.npy_file}'
    if osp.isdir(npy_file):
        npy_path_list = [osp.join(npy_file, name) for name in os.listdir(npy_file)
                         if name.endswith('.npy')]
    else:
        npy_path_list = [npy_file]
    for npy_path in npy_path_list:
        cvt_npy2seg(npy_path, args.out)
