import os
import glob
import struct
import argparse
import concurrent.futures
import numpy as np


DEFAULT_SEQS = [
    '2012-02-18',
    '2012-02-19',
    '2012-03-31',
    '2012-05-11',
    '2012-02-12',
    '2012-03-17',
    '2012-03-25',
    '2012-05-26',
]


def convert_nclt(x_s, y_s, z_s):
    scaling = 0.005  # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def load_velodyne_binary_nclt(filename):
    hits = []
    with open(filename, "rb") as f_bin:
        while True:
            x_str = f_bin.read(2)
            if x_str == b'':  # eof
                break
            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            _ = struct.unpack('B', f_bin.read(1))[0]

            x, y, z = convert_nclt(x, y, z)

            hits += [[x, y, z, i]]

    hits = np.array(hits)

    return hits


def convert_file(input_file, output_dir):
    scan = load_velodyne_binary_nclt(input_file).astype(np.float32)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    scan.tofile(output_file)
    return output_file


def processing(data_root, seq):
    input_dir = os.path.join(data_root, seq, 'velodyne_sync')
    output_dir = os.path.join(data_root, seq, 'velodyne_left')

    if not os.path.isdir(input_dir):
        print(f'Skip {seq}: input directory not found: {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, '*.bin')))
    if not file_list:
        print(f'Skip {seq}: no .bin files found in {input_dir}')
        return

    print(f'Processing {seq}: {len(file_list)} files')
    for file in file_list:
        convert_file(file, output_dir)


def processing_parallel(data_root, seq, workers):
    input_dir = os.path.join(data_root, seq, 'velodyne_sync')
    output_dir = os.path.join(data_root, seq, 'velodyne_left')

    if not os.path.isdir(input_dir):
        print(f'Skip {seq}: input directory not found: {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, '*.bin')))
    if not file_list:
        print(f'Skip {seq}: no .bin files found in {input_dir}')
        return

    print(f'Processing {seq}: {len(file_list)} files with {workers} workers')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(convert_file, input_file, output_dir)
            for input_file in file_list
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if index % 100 == 0 or index == len(futures):
                print(f'Processed {seq}: {index}/{len(futures)} files')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert NCLT velodyne_sync raw binary files to LightLoc velodyne_left float32 files.'
    )
    parser.add_argument(
        '--data_root',
        required=True,
        help='Path to the NCLT dataset root, e.g. /data/NCLT',
    )
    parser.add_argument(
        '--seqs',
        nargs='+',
        default=DEFAULT_SEQS,
        help='NCLT sequence names to process.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel worker processes. Use 1 for sequential processing.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    workers = max(1, args.workers)
    for seq in args.seqs:
        if workers == 1:
            processing(data_root, seq)
        else:
            processing_parallel(data_root, seq, workers)