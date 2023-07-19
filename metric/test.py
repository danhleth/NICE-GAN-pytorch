import os
import cv2
from ssim import SSIM
from psnr import PSNR

metric_ssim = SSIM()
metric_psnr = PSNR()

real_path = "testB/5.jpg"
generated_path = "fakeA/5.jpg"

real_img = cv2.imread(real_path)
gen_img = cv2.imread(generated_path)

real_img = cv2.resize(real_img, (256,256), interpolation=cv2.INTER_LINEAR)
gen_img = cv2.resize(gen_img, (256,256), interpolation=cv2.INTER_LINEAR)

print("psnr: ", metric_psnr(gen_img, real_img))
print("ssim: ", metric_ssim(gen_img, real_img))