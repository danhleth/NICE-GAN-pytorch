import torch
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from glob import glob
import shutil


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
NII_2D_IMAGE_EXTENSION = [".npy"]

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


class DatasetFolder(torch.utils.data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        # sample = sample / 255.
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(image=target)["image"]
        return sample, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def cv2_loader(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    min_value = image.min()
    if min_value < 0:
        print(path)
    return image

def np_loader(path):
    image = np.load(path)
    return image

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=cv2_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

class NiiImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=np_loader):
        super(NiiImageFolder, self).__init__(root, loader, NII_2D_IMAGE_EXTENSION,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


def remove_file_in_dir(directory):
    if not os.path.exists(directory):
        return
    files = [os.path.join(directory, x) for x in os.listdir(directory)]
    for file_path in files:
        os.remove(file_path)


def copy_files(source_dir, source_pattern="*_t2_*.jpg", dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset/brats_t2_to_flair_npy/test_source"):
    os.makedirs(dst_dir, exist_ok=True)
    
    source_imgs = glob(os.path.join(source_dir,source_pattern))
    
    for i, image_path in tqdm(enumerate(source_imgs)):
        image_name = image_path.split('/')[-1]
        shutil.copy(image_path, f"{dst_dir}/{image_name}")


def copy_files_new(source_dir, dataset_type="train", pattern="t1ce",dst_dir=""):
    source_dir = Path(source_dir)
    source_dir = source_dir/dataset_type

    for patient_dir in tqdm(source_dir.iterdir()):
        patient_dir = source_dir/patient_dir/pattern
        for slice_path in patient_dir.iterdir():
            slice_name = slice_path.name
            shutil.copy(slice_path, f"{dst_dir}/{slice_name}")


if __name__=='__main__':
    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="val",
                   pattern="flair",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/train_target")
    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="val",
                   pattern="t2",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/train_source")
    
    
    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="train",
                   pattern="flair",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/train_target")
    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="train",
                   pattern="t2",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/train_source")


    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="val",
                   pattern="flair",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/test_target")
    copy_files_new(source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new",
                   dataset_type="val",
                   pattern="t2",
                   dst_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/brats_t2_to_flair_nii/test_source")

