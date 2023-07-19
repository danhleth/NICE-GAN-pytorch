import cv2
import argparse
from pathlib import Path
import numpy as np

from tqdm import tqdm
import multiprocessing
from skimage.metrics import structural_similarity as ssim
from skimage.util.dtype import dtype_range

def parse_args():
    parser = argparse.ArgumentParser(description='Similarity synthesis')
    parser.add_argument('--path_real', type=str,
                        help='path to real data')
    parser.add_argument('--path_gen', type=str,
                        help='path to generated data')

    args = parser.parse_args()
    return args

def cv2_loader(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def preprocess(image, width=256, height=256):
    out_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return out_img

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    if imageA.shape[-1] == 1:
        imageA = np.squeeze(imageA, axis=-1)
        imageB = np.squeeze(imageB, axis=-1)

    mse_score = mse(imageA, imageB)
    ssim_score = ssim(imageA, imageB)
    psnr_score = calculate_psnr(imageA, imageB)
    print(type(ssim_score), type(psnr_score), type(mse_score))
    return ssim_score, psnr_score, mse_score

def evaluate_by_dir(dir_real, dir_gen, pattern="", dataset_name="brats_t2_to_flair_npy", dataset_type="", save_log=False):
    dir_real = Path(dir_real)
    dir_gen = Path(dir_gen)
    pattern="*" if len(pattern) == 0 else f"*_{pattern}_*.jpg"
    p = dir_real.glob(pattern)
    ssim_score = 0.0
    psnr_score = 0.0
    mse_score = 0.0


    len_p = 0
    for real_img_path in tqdm(p, desc=pattern):
        len_p += 1
        real_img_path = str(real_img_path)
        filename = real_img_path.split('/')[-1]
        try:
            image_real = cv2_loader(image_path=str(real_img_path))
            image_real = preprocess(image_real)

            gen_img_path = dir_gen/filename
            image_gen = cv2_loader(str(gen_img_path))
            image_gen = preprocess(image_gen)
        except:
            print("real_image_path: ", str(real_img_path))
            print("gen_image_path: ", str(gen_img_path))
            exit(0)

        tmp_ssim, tmp_psnr, tmp_mse = compare_images(image_real, image_gen)
        ssim_score += tmp_ssim
        mse_score += tmp_mse
        psnr_score += tmp_psnr
    
    ssim_score = ssim_score/len_p
    psnr_score = psnr_score/len_p
    mse_score = mse_score/len_p
    if save_log:
        with open(f"evaluate_{dataset_name}_{dataset_type}.txt", 'a') as f:
            f.writelines(f"evaluate follow pattern: {pattern} with len_p: {len_p}\n")
            f.writelines(f"psnr: {psnr_score}  \t  ssim: {ssim_score}  \t  mse: {mse_score}\n")
            f.writelines("-----------\n")

    print(f"evaluate follow pattern: {pattern} with len_p: {len_p}")
    print(f"psnr: {psnr_score}  \t  ssim: {ssim_score}  \t  mse: {mse_score}\n")
    print("----------")
        


if __name__=="__main__":
    # args = parse_args()
    
    # path_real = Path(args.path_real)
    # path_gen = Path(args.path_gen)
    dataset_name = "brats_t1_to_t1ce_npy"
    dataset_type="train"
    path_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target"
    path_gen=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/{dataset_name}/{dataset_type}/image"
    
    default_num_processes = 8
    with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
        evaluate_by_dir(path_real, path_gen, pattern="axial", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        evaluate_by_dir(path_real, path_gen, pattern="coronal", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        evaluate_by_dir(path_real, path_gen, pattern="sagittal", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        # evaluate_by_dir(path_real, path_gen, pattern="", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)

    dataset_name = "brats_t2_to_flair_npy"
    path_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target"
    path_gen=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/{dataset_name}/{dataset_type}/image"
    with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
        evaluate_by_dir(path_real, path_gen, pattern="axial", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        evaluate_by_dir(path_real, path_gen, pattern="coronal", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        evaluate_by_dir(path_real, path_gen, pattern="sagittal", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
        # evaluate_by_dir(path_real, path_gen, pattern="", dataset_name=dataset_name, dataset_type=dataset_type, save_log=True)
