import torch
import os
from tqdm import tqdm
import json
import nibabel as nib
import multiprocessing
import numpy as np

IMG_EXTENSIONS = ['.nii.gz']

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
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
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


def nibabel_loader(path):
    data = nib.load(path)
    data = data.get_fdata()
    return data


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=nibabel_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

def numpy_2_nifti(npy_data):
    image = nib.Nifti1Image(npy_data,affine=np.eye(4))
    return image

def prepare_dataset_to_dir(save=True):
    dataset_json = "/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/splits_final.json"
    root_dir = "/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/nnUNetPlans_3d_fullres"
    save_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/brats_3d_nnUnet_preprocessed"
    with open(dataset_json, 'r') as f:
        js = json.load(f)
    js = js[0]
    map_modalities_dict = {0: "t1", 1: "t1ce", 2: "t2", 3: "flair"} # who knows the order?
    with multiprocessing.get_context("spawn").Pool(8) as segmentation_export_pool:
        for dataset_type, list_files in js.items():
            os.makedirs(f"{save_dir}/{dataset_type}", exist_ok=True)
            for data_name in tqdm(list_files):
                npy = np.load(f"{root_dir}/{data_name}.npy")
                for modalities_ix, modalities in enumerate(npy):
                    nib_image = numpy_2_nifti(modalities)
                    print("preapre save at: ", f"{save_dir}/{dataset_type}/{data_name}_{map_modalities_dict[modalities_ix]}.nii.gz")
                    nib.save(nib_image, f"{save_dir}/{dataset_type}/{data_name}_{map_modalities_dict[modalities_ix]}.nii.gz")

if __name__=='__main__':
    # prepare_dataset_to_dir()
    pass