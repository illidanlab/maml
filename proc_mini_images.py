"""
Script for converting from csv file datafiles to a directory for each image (which is how it is loaded by MAML code)

Acquire miniImagenet from Ravi & Larochelle '17, along with the train, val, and test csv files. Put the
csv files in the miniImagenet directory and put the images in the directory 'miniImagenet/images/'.
Then run this script from the miniImagenet directory:
    cd data/miniImagenet/
    python proc_images.py
"""

from __future__ import print_function

import csv
import glob
import os
import multiprocessing
from PIL import Image

def resize_batch_images(images_files, batch_id):
    # Resize images
    for i, image_file in enumerate(images_files):
        im=Image.open(image_file)
        im=im.resize((84, 84), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 500 == 0:
            print("processing {}-bath, {}-th images".format(batch_id, i))

def move_batch_images(datatype, config):
    file_path=os.path.join(config['data_path'], 'miniImagenet', '{}.csv'.format(datatype))
    with open(file_path, 'r') as f:
        reader=csv.reader(f, delimiter=',')
        last_label=''
        for i, row in enumerate(reader):
            if i == 0:  # skip the headers
                continue
            if i % 500 == 0:
                print("Moving {}-th batch of {} images.".format(i, datatype))
            image_name, label=row[0], row[1]
            if label != last_label:
                cur_dir=os.path.join(config['data_path'], 'miniImagenet', datatype, label)
                cmd='mkdir -p {}'.format(cur_dir)
                os.system(cmd)
                last_label=label
            old_path=os.path.join(config['data_path'], 'miniImagenet', 'images', image_name)
            mv_cmd='mv {} {}'.format(old_path, cur_dir)
            os.system(mv_cmd)

def process_images(config={}, n_threads=10):
    path_to_images = os.path.join(config['data_path'], 'miniImagenet/images/')
    assert os.path.isdir(path_to_images)

    all_images=glob.glob(path_to_images + '*')
    thread_jobs = []
    batch_per_thread = len(all_images) // (n_threads - 1)
    for i in range(n_threads):
        start = i * batch_per_thread
        end = min((i+1) * batch_per_thread, len(all_images))
        if start >= end: continue
        batch_files = all_images[start:end]
        p = multiprocessing.Process(target=resize_batch_images, args=(batch_files, i))
        thread_jobs.append(p)
        p.start()

    for p in thread_jobs:
        p.join()

    # Put in correct directory
    print("Begin to put images to individual folders ...")
    thread_jobs = []
    for datatype in ['train', 'val', 'test']:
        p = multiprocessing.Process(target=move_batch_images, args=(datatype, config))
        thread_jobs.append(p)
        p.start()

    for p in thread_jobs:
        p.join()

