import os
import shutil
from pathlib import Path


def parse_filename(filename):
    no_ext = filename[:-4]
    p1 = no_ext.find('_')
    p2 = no_ext.find('_', p1 + 1)
    return filename[:p2 + 1], filename[p2+1:], filename[-4:]


ROOT_DIR = './SRAD2018_Test_1'
dirs = []
with os.scandir(ROOT_DIR) as it:
    for entry in it:
        if entry.is_dir():
            dirs.append(entry)

for i, d in enumerate(dirs):
    files = []
    with os.scandir(d.path) as it:
        for f in it:
            if f.is_file():
                files.append(f)
    files = sorted(files, key=lambda x:x.name)
    last_file = files[-1]
    prefix, _, ext = parse_filename(last_file.name)

    for j in range(1, 7):
        format_i = 'f{:03d}'.format(j)
        result_filename = prefix + format_i + ext
        result_path = Path(os.path.join('./result_baseline', d.name, result_filename))
        if os.path.exists(result_path.parent) is not True:
            os.mkdir(result_path.parent)

        shutil.copy2(f.path, result_path)
    if i % 100 == 0:
        print('process: {0}/100.', i/100)

