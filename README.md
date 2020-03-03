## NICE-GAN &mdash; Official PyTorch Implementation
### : Reusing Discriminators for Encoding Towards Unsupervised Image-to-Image Translation

<div align="center">
  <img src="./figure/title_figure.png" width = '586px' height = '367px'>
</div>


### [Paper](https://arxiv.org/abs/2003.00273) 

> **Reusing Discriminators for Encoding Towards Unsupervised Image-to-Image Translation**<br>
> Runfa Chen, Wenbing Huang, Binghui Huang, Fuchun Sun, Bin Fang (Tsinghua University)
>
> **Abstract** *Unsupervised image-to-image translation is a central task in computer vision. Current translation frameworks will abandon the discriminator once the training process is completed. This paper contends a novel role of the discriminator by reusing it for encoding the images of the target domain. The proposed architecture, termed as NICE-GAN, exhibits two advantageous patterns over previous approaches: First, it is more compact since no independent encoding component is required; Second, this plug-in encoder is directly trained by the adversary loss, making it more informative and trained more effectively if a multi-scale discriminator is applied. The main issue in NICE-GAN is the coupling of translation with discrimination along the encoder, which could incur training inconsistency when we play the min-max game via GAN. To tackle this issue, we develop a decoupled training strategy by which the encoder is only trained when maximizing the adversary loss while keeping frozen otherwise. Extensive experiments on four popular benchmarks demonstrate the superior performance of NICE-GAN over state-of-the-art methods in terms of FID, KID, and also human preference. Comprehensive ablation studies are also carried out to isolate the validity of each proposed component.*

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```
### Train
```
> python main.py --dataset cat2dog
```
* If the memory of gpu is **not sufficient**, set `--light` to True

### Test
```
> python main.py --dataset cat2dog --phase test
```

### Metric
```
> python fid_kid.py testA fakeA --mmd-var 
```
* You can use gpu, set `--gpu` to **the index of gpu**, such as `--gpu 0`

## Network
<div align="center">
  <img src = './figure/network.png' width = '929px' height = '310px'>
</div>

## Comparison
<div align="center">
  <img src = './figure/contrast.png' width = '752px' height = '868px'>
</div>

## User study
<div align="center">
  <img src = './figure/column_big.png' width = '448px' height = '238px'>
</div>

## t-SNE
<div align="center">
  <img src = './figure/tsne.png' width = '711px' height = '207px'>
</div>

## Heatmaps
<div align="center">
  <img src = './figure/heatmaps.png' width = '643px' height = '456px'>
</div>

## Shared latent space
<div align="center">
  <img src = './figure/shared_int.png' width = '890px' height = '247px'>
</div>

## Citation
If you find this code useful for your research, please cite our paper:

```
@misc{chen2020reusing,
    title={Reusing Discriminators for Encoding Towards Unsupervised Image-to-Image Translation},
    author={Runfa Chen and Wenbing Huang and Binghui Huang and Fuchun Sun and Bin Fang},
    year={2020},
    eprint={2003.00273},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Author
Runfa Chen, Wenbing Huang, Binghui Huang, Fuchun Sun, Bin Fang (Tsinghua University)
[Tsinghua Robot Learning Lab](https://github.com/tsinghua-rll)
