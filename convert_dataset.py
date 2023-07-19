import nibabel as nib
import numpy as np
from glob import glob
import os
import cv2
from tqdm import tqdm
from scipy.ndimage import median_filter
from pathlib import Path
import shutil

def cv_loader(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return img


def numpy_loader(img_path):
    img = np.load(img_path)
    return img


def stack_image(pattern, loader=cv_loader):
    files = glob(pattern)
    stack = []
    for image_path in files:
        img = loader(image_path)
        stack.append(img)
    return np.array(stack)


def numpy_2_nifti(npy_data, header=None, transform=None):
    if transform is None:
        transform = np.eye(4)
    
    if header:
        image = nib.Nifti1Image(npy_data,affine=transform, header=header)
    
    image = nib.Nifti1Image(npy_data,affine=transform)
    return image


def load_nib_2_npy(nifti_path):
    data = nib.load(nifti_path)
    data = data.get_fdata()
    return data


def reformat_data(generated_dir, dataset, save_to_dir="brats_generated_data", axis="sagittal", settype="train"):
    target_name = dataset.split('_')[-2]
    npy_save_dir = f"{save_to_dir}/{axis}/npy"
    nifti_save_dir = f"{save_to_dir}/{axis}/NIfTI"
    os.makedirs(save_to_dir, exist_ok=True)
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(nifti_save_dir, exist_ok=True)
    
    
    files = glob(f"{generated_dir}/{dataset}/{settype}/image/*_{axis}_0.jpg")
    print(f"[*] Processing axis: {axis}")
    for i, file_path in tqdm(enumerate(files), desc=f"{dataset}"):
        file_name = file_path.split('/')[-1].split(target_name)[0] + target_name
        file_pattern = file_path[:-5]+"*"
        data = stack_image(file_pattern)
        np.save(f"{npy_save_dir}/{file_name}.npy", data)
        data = numpy_2_nifti(data)
        nib.save(data,f"{nifti_save_dir}/{file_name}.nii.gz")


def get_metadata_from_nifti(file_pattern, axis="axial", get_header_from="t1"):
    data_origin = "/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/brats2018_origin"
    filename = file_pattern.split('/')[-1].split(axis)[0][:-1]
    database = glob(f"{data_origin}/**/{filename}")
    file_path = os.path.join(database[0], f"{filename}_{get_header_from}.nii.gz")
    header, transform = None, None
    # print(file_path)
    data = nib.load(file_path)
    header = data.header.copy()
    transform = data.affine.copy()
    data = data.get_fdata()
    origin_shape = data.shape
    return header, transform, origin_shape


def reformat_data_npy(generated_dir, dataset, save_to_dir="brats_generated_data", axis="sagittal", settype="train"):
    target_name = dataset.split('_')[-2]
    npy_save_dir = f"{save_to_dir}/{axis}/npy"
    nifti_save_dir = f"{save_to_dir}/{axis}/NIfTI"
    os.makedirs(save_to_dir, exist_ok=True)
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(nifti_save_dir, exist_ok=True)
    
    files = glob(f"{generated_dir}/{dataset}/{settype}/generated_npy/*_{axis}_0.npy")
    print(f"[*] Processing axis: {axis}")

    for i, file_path in tqdm(enumerate(files), desc=f"{dataset}"):
        file_name = file_path.split('/')[-1].split(axis)[0] + target_name
        file_pattern = file_path[:-5]+"*"
        data = stack_image(file_pattern, loader=numpy_loader)
        np.save(f"{npy_save_dir}/{file_name}.npy", data)
        header, transform, origin_shape = get_metadata_from_nifti(file_path)
        data = np.resize(data, origin_shape)
        print(data.shape)
        data = numpy_2_nifti(data, header=header, transform=transform)
        nib.save(data,f"{nifti_save_dir}/{file_name}.nii.gz")


def copy_header_and_transform(source_file, target_file):
    source_img = nib.load(source_file)
    hdr = source_img.header.copy()
    aff = source_img.affine.copy()
    target_img = nib.load(target_file)
    target_img = nib.Nifti1Image(np.asanyarray(target_img.dataobj), affine=aff, header=hdr)
    nib.save(target_img, target_file)


def de_artifact_gan(axis_dir):
    files = os.listdir(f"{axis_dir}/NIfTI/")
    nifti_save_dir = f"{axis_dir}/NIfTI_median_filter"
    os.makedirs(nifti_save_dir, exist_ok=True)
    for i, file_name in tqdm(enumerate(files), desc=f"deartifact axis {axis_dir.split('/')[-1]}"):
        data = nib.load(os.path.join(axis_dir,"NIfTI",file_name))
        try:
            data = data.get_fdata()
        except Exception as e:
            print(file_name)
        
        data = median_filter(data, size=7, mode="reflect")
        data = numpy_2_nifti(data)
        nib.save(data,f"{nifti_save_dir}/{file_name}")


def get_files_id(dataset_type="train"):
    if dataset_type == "test":
        candidate = os.listdir("/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/test")
        data = []
        for c in candidate:
            if not ".csv" in c:
                data.append(c)
        return data
    import json
    with open("/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/splits_final.json", 'r') as f:
        data = json.load(f)
    data = data[0]
    return data[dataset_type]


def de_artifact_gan_average_axis(dataset_dir, dataset_type="train", modalities=["t1ce", "flair"]):
    print(dataset_type)
    candidates = get_files_id(dataset_type)
    print(candidates)
    exit(0)
    save_dir = f"{dataset_dir}/{dataset_type}/3_axis"
    os.makedirs(save_dir, exist_ok=True)

    for modal in modalities:
        for candidate in candidates:
            axial_path = os.path.join(dataset_dir, dataset_type, "axial", "NIfTI", f"{candidate}_{modal}.nii.gz")
            coronal_path = os.path.join(dataset_dir, dataset_type, "coronal", "NIfTI", f"{candidate}_{modal}.nii.gz")
            sagittal_path = os.path.join(dataset_dir, dataset_type, "sagittal", "NIfTI", f"{candidate}_{modal}.nii.gz")

            axial = load_nib_2_npy(axial_path)
            coronal = load_nib_2_npy(coronal_path)
            sagittal = load_nib_2_npy(sagittal_path)
            
            final = (axial*0.1 + coronal*0.1 + 0.8*sagittal)
            final = numpy_2_nifti(final)
            nib.save(final, f"{save_dir}/{candidate}_{modal}.nii.gz")


def copy_data_to_nnunet(source_dir, dst_ds_nnunet_dir, dataset_type="train"):
    map_dict = {"t1": "0000", "t1ce": "0001", "t2": "0002", "flair": "0003"}
    source_dir = Path(source_dir)
    if dataset_type == "test":
        dst_ds_nnunet_dir = Path(dst_ds_nnunet_dir)/"imagesTs"
    elif dataset_type == "train" or dataset_type == "val":
        dst_ds_nnunet_dir = Path(dst_ds_nnunet_dir)/"imagesTr"
    print("image will save to: ", dst_ds_nnunet_dir)
    for source_path in tqdm(source_dir.iterdir()):
        filename = source_path.name
        key = filename.split(".")[0].split("_")[-1]
        filename = filename.replace(key, map_dict[key])
        destination_path = dst_ds_nnunet_dir/filename
        # print(destination_path)
        shutil.copy(source_path, destination_path)
    

if __name__ == "__main__":
    de_artifact_gan_average_axis(dataset_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/train/axial")
    pass