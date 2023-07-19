import torch

from PIL import Image
import nibabel as nib

import os
import os.path
import numpy as np
import pandas as pd 


from glob import glob


def slice_through_file_brain(file_path, save_dir):
    scan = nib.load(file_path)
    scan = scan.get_fdata()
    for plane in range(scan.shape[2]):
        p = scan[:,:,plane].astype(np.uint8)
        img = Image.fromarray(p)
        save_name = file_path.split('/')[-1]
        save_name = save_name.split(".")[0]
        save_name = f"{plane}_{save_name}.jpg"
        save_path = os.path.join(save_dir, save_name)
        img.save(save_path)



def create_pair_file(source_dir, target_dir):
    source_imgs = os.listdir(source_dir)
    target_imgs = os.listdir(target_dir)
    print("len_source: ", len(source_imgs))
    print("len_targets: ", len(target_imgs))

def write_annotation(content, file, mode='w'):
    with open(file,mode) as f:
        f.write(content)

def gen_annotation_A_less_B(source, target):
    """
        source is less than target
    """
    ratio = target.shape[2] / source.shape[2]
    


def gen_annotation_one_one(source_path, target_path):
    scan_src = nib.load(source_path)
    scan_src = scan_src.get_fdata()

    scan_target = nib.load(target_path)
    scan_target = scan_target.get_fdata()

    if scan_target.shape[2] > scan_src.shape[2]:
        scan_ratio = 1
    elif scan_target.shape[2] < scan_src.shape[2]:
        scan_ratio = 2
    else:
        scan_ratio = 3


def gen_annotation_one_many(source_file_path, target_dir):
    for target_path in target_dir:
        annotation = gen_annotation_one_one(source_file_path, target_path)


def convert_2d_img(root_dir):
    source = f"{root_dir}/T1ce_imgs"
    target = f"{root_dir}/T2_imgs"
    os.makedirs(source, exist_ok=True)
    os.makedirs(target, exist_ok=True)

    t1ce_fies = glob(f"{root_dir}/training_source/*_ceT1.nii.gz")
    t2_files = glob(f"{root_dir}/training_target/*.nii.gz")
    
    for file_path in t2_files:
        slice_through_file_brain(file_path, target)

    for file_path in t1ce_fies:
        slice_through_file_brain(file_path, source)


class CrossMODADataset(torch.utils.data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    # convert_2d_img("/home/dtpthao/workspace/brats_projects/domainadaptation/datasets")
    # create_pair_file(source_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/T1ce_imgs",
    #                 target_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/T2_imgs")

    gen_annotation_one_one(source_path='/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/training_source/crossmoda2021_ldn_1_ceT1.nii.gz',
                           target_path='/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/training_target/crossmoda2021_ldn_106_hrT2.nii.gz')
    


