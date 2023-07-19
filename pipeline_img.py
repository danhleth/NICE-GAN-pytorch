from dataset.brats_2d_dataset import copy_files, remove_file_in_dir
from main import parse_args
from models.NICE import NICE
from convert_dataset import reformat_data, de_artifact_gan, de_artifact_gan_average_axis
import multiprocessing

from metric.brats import evaluate_by_dir

args = parse_args()
default_num_processes=4
source_copy_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/datasets/brats_slices_full"
generated_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data"
dataset_type="test"


# dataset_name="brats_t1_to_t1ce_norm_inplace"
# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/train_source")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#             source_pattern="*_t1_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/train_source")


# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/train_target")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#             source_pattern="*_t1ce_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/train_target")

# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#             source_pattern="*_t1_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")


# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#             source_pattern="*_t1ce_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")

# args.dataset=dataset_name
# args.device="cuda:1"
# gan = NICE(args)
# gan.build_model()
# gan.test_test(generated_dir, dataset_type)


# with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="axial",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="coronal",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="sagittal",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)

# for axis in ["axial", "coronal", "sagittal"]:
#     reformat_data(generated_dir, 
#                 dataset_name, 
#                 save_to_dir=f"{generated_dir}/{dataset_type}",
#                 axis=axis,
#                 settype=dataset_type)

dataset_name="brats_t2_to_flair_npy"
# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#             source_pattern="*_t2_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_source")

# remove_file_in_dir(directory=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")
# copy_files(source_dir=f"{source_copy_dir}/{dataset_type}",
#         source_pattern="*_flair_*.jpg",
#             dst_dir=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target")

# args.dataset=dataset_name
# args.device="cuda:1"
# gan = NICE(args)
# gan.build_model()
# gan.test_test(generated_dir,dataset_type)

# with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="axial",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="coronal",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)
#     evaluate_by_dir(dir_real=f"/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/dataset_dir/{dataset_name}/test_target",
#                     dir_gen=f"{generated_dir}/{dataset_name}/{dataset_type}/image",
#                     pattern="sagittal",
#                     dataset_name=dataset_name,
#                     dataset_type=dataset_type,
#                     save_log=True)

for axis in ["axial", "coronal", "sagittal"]:
    reformat_data(generated_dir, 
                dataset_name, 
                save_to_dir=f"{generated_dir}/{dataset_type}",
                axis=axis,
                settype=dataset_type)

# de_artifact_gan_average_axis(dataset_dir="/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data",
#                              dataset_type=dataset_type,
#                              modalities=["t1ce", "flair"])

# for axis in ["axial", "coronal", "sagittal"]:
#     de_artifact_gan(axis_dir=f"{generated_dir}/{dataset_type}/{axis}")
