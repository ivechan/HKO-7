import os
import re
import random
import time
import numpy as np

from imageio import imread
from pathlib import Path
from random import choice
random.seed() # the system time is used for seed.


class ICDMIterator(object):

    """ICDM Iterator"""

    def __init__(self, root_dir, max_index=60):
        """
        :root_dir: TODO

        """
        assert os.path.exists(root_dir)
        self._root_dir = root_dir
        self._subdirs = self._scan_dir(self._root_dir)
        print('loading dataset, the total num of subdirs:', len(self._subdirs))

    def _scan_dir(self, dirpath, is_sort=False):
        filepaths = []
        with os.scandir(dirpath) as it:
            for entry in it:
                if not entry.name.startswith('.'):
                    filepaths.append(entry)

        if is_sort:
            filepaths = sorted(filepaths, key=lambda x: x.name)
        return filepaths

    def _load_frames(self, subdir, start, in_len, out_len):
        def format_it(i):
            return "{:03d}".format(i)

        inputs = []
        outputs = []
        for i in range(start, start+in_len):
            inputs.append(os.path.join(subdir, subdir.name + '_'+ format_it(i) + '.png'))

        for i in range(start+in_len, start+in_len+out_len):
            outputs.append(os.path.join(subdir, subdir.name + '_'+ format_it(i) + '.png'))
        input_frames = []
        output_frames = []
        for file in inputs:
            input_frames.append(imread(file))

        for file in outputs:
            output_frames.append(imread(file))

        return np.asarray(input_frames), np.asarray(output_frames)

    def sample(self, batch_size=1, in_len=6, out_len=6):
        """
        :batch_size = 1
        """
        input_batches = []
        output_batches = []
        for _ in range(batch_size):
            subdir = choice(self._subdirs)
            start = choice(range(0, 61 - out_len - in_len))
            inputs, outputs = self._load_frames(subdir, start, in_len,
                                                out_len)
            # 增加channel维度
            inputs = np.expand_dims(inputs, axis=1)
            outputs = np.expand_dims(outputs, axis=1)
            input_batches.append(inputs)
            output_batches.append(outputs)
        input_batches = np.asarray(input_batches)
        output_batches = np.asarray(output_batches)
        # NTCWH
        # Transposed to TNCWH
        input_batches = np.transpose(input_batches, (1, 0, 2, 3, 4))
        output_batches = np.transpose(output_batches, (1, 0, 2, 3, 4))
        # print(input_batches.shape, output_batches.shape)
        return input_batches + 1, output_batches + 1 #tricky


if __name__ == "__main__":
    icdm_iter = ICDMIterator(root_dir=r'C:\Users\jing\projects\nowcasting\HKO-7\icdm_data\SRAD2018_TRAIN_010')
    i, o = icdm_iter.sample(batch_size=3)
    print(i.shape, o.shape)

    
