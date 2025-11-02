# Adversarial Detection Avoidance Attacks

## About

This repository contains the source code for creating a deduplicated version of the ImageNet ILSVRC 2012 challenge [dataset](https://www.image-net.org/challenges/LSVRC/2012/), used in the USENIX Security ‘22 paper [Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning](https://arxiv.org/abs/2106.09820) by Shubham Jain*, Ana-Maria Cretu* and Yves-Alexandre de Montjoye.

We have used this deduplicated dataset to evaluate the false positives of perceptual hashing-based client-side scanning systems. Doing this on a dataset with duplicates (such as ImageNet) would be incorrect as it would lead to overestimating false positives. Indeed, false positives are evaluated by querying a database (in our experiments, a subset of ImageNet) with a large number of images (sampled from a disjoint subset of ImageNet). If there are duplicates between the database and the query images, we would be wrongly counting true positive pairs (i.e., pairs of duplicates) as false positives.

We release this dataset to allow other researchers to evaluate false positives on other perceptual hashing algorithms, databases, and future client-side scanning system designs. Other potential uses of the deduplicated dataset include the creation of new train, validation and test splits which are truly disjoint (we indeed found duplicates between the train, validation and test splits of the original dataset).

## Deduplication details

For privacy and ethical reasons, we first removed images containing faces from ImageNet using the list provided by the paper [A Study of Face Obfuscation in ImageNet](https://arxiv.org/abs/2103.06191https://github.com/princetonvisualai/imagenet-face-obfuscation/blob/main/README.md) in the file `face_annotations_ILSVRC.json`. We then removed the exact and near-duplicate images through a semi-automated analysis of the dataset, detailed in Appendix B of our paper. Thus, we have only deduplicated the ImageNet dataset that does not contain any faces. The deduplicated dataset should contain 1,179,448 images with 8526 duplicates removed.

## How to use

### Step 1
Download the ImageNet dataset from ILSVRC 2012. Download the training images for Task 1 and Task 2, and all the validation and test images. Extract these images in the `datasets/imagenet/` folder in the repository. The images can be extracted as `datasets/imagenet/train/`, `datasets/imagenet/val/` and `datasets/imagenet/train/` for simplicity. The dataloader looks for all images stored in the folder and subfolders. The files can be extracted using `process_imagenet.py`. To run this, modify the data paths in the file and run:
```
python process_imagenet.py
```

### Step 2

Load the dataset using the dataloader (`dataloader.py`). The dataloader loads all the images then discards (1) all images containing faces provided in the `face_annotations_ILSVRC.json` file and (2) all the duplicates provided in `duplicate_images.json`.



## How to cite

If you use this dataset, please cite our work:

```
@inproceedings{jain2022adversarial,
  title={Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning},
  author={Jain, Shubham and Crețu, Ana-Maria and de Montjoye, Yves-Alexandre},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  pages={2317--2334},
  year={2022}
}
```
