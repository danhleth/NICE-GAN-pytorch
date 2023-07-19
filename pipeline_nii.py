from dataset.brats_2d_dataset import copy_files_new, remove_file_in_dir
from main import parse_args
from models.NICE import NICE
from convert_dataset import (reformat_data_npy, 
                            de_artifact_gan, 
                            de_artifact_gan_average_axis, 
                            copy_data_to_nnunet)
import multiprocessing

from metric.brats import evaluate_by_dir

args = parse_args()
default_num_processes=4
source_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new"
generated_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data"
dataset_type="test"


dataset_name="brats_t2_to_flair_nii"
# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")
# copy_files_new(source_dir=source_dir,
#             dataset_type=dataset_type,
#             pattern="t2",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")


# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")
# copy_files_new(source_dir=source_dir,
#             dataset_type=dataset_type,
#             pattern="flair",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")

# args.dataset=dataset_name
# args.device="cuda:1"
# gan = NICE(args)
# gan.build_model()
# gan.test_test(generated_dir, dataset_type)

# for axis in ["axial"]:
#     reformat_data_npy(generated_dir, 
#                 dataset_name, 
#                 save_to_dir=f"{generated_dir}/{dataset_type}",
#                 axis=axis,
#                 settype=dataset_type)


copy_data_to_nnunet(source_dir=f"{generated_dir}/{dataset_type}/axial/NIfTI",
                    dst_ds_nnunet_dir="/home/dtpthao/workspace/nnUNet/env/raw/Dataset034_BraTS2018_nicegan",
                    dataset_type=dataset_type)
