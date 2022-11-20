# Relational based Transfer Learning for Automatic Optical Inspection based on domain discrepancy

<a href="https://colab.research.google.com/drive/1qw5F_V8FH2yorPPX8H6_BKFIPiqhFyB3?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Abstract
Transfer learning is a promising method for AOI applications since it can significantly shorten sample collection time and improve efficiency in today’s smart manufacturing. However, related research enhanced the network models by applying TL without considering the domain similarity among datasets, the data long-tailedness of a source dataset, and mainly used linear transformations to mitigate the lack of samples. This research applies relational-based TL via domain similarity to improve the overall performance and data augmentation in both target and source domains to enrich the data quality and reduce the imbalance. Given a group of source datasets from similar industrial processes, we define which group is the most related to the target through the domain discrepancy score and the number of samples each has. Then, we transfer the chosen pre-trained backbone weights to train and fine-tune the target network. Our research suggests increases in the F1 score and the PR curve up to 20% compared with TL using benchmark datasets.

**Keywords**: machine vision, automatic optical inspection (AOI), transfer learning, domain similarity, data augmentation, supervised learning, domain adaption.

## Requirements

- If you plan to run inferences in your local machine using YOLOv5, please follow the instructions provided by [Ultralytics](https://github.com/ultralytics/yolov5)

- Else, please write the commands below:

```
conda create -y --name RTLAOI-DD --file requirements.txt 
conda activate RTLAOI-DD
pip install thop>=0.1.1 opencv-python>=4.1.1
```

You can also find all notebooks available in Google Colab. Please click on them, and you will find a link to see them.

## Description
This repository complements the research paper presented in [SPIE COS Phonotics Asia 2022](https://spie.org/spie-cos-photonics-asia/presentation/Relational-based-transfer-learning-for-automatic-optical-inspection-based-on/12317-42?SSO=1)

We included a short version of the full-sized dataset for your reference. Unfortunately, we cannot share the full-size dataset, so the results obtained using it may not fit the same as the ones presented in the paper.

### Domain_similarity.ipynb
This Jupyter notebook contains a demonstration of how we calculated the Earth Mover’s Distance for domain discrepancy estimation from stratch. It is available in Google Colab too.

<a href="https://colab.research.google.com/drive/1qw5F_V8FH2yorPPX8H6_BKFIPiqhFyB3?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Weights & Biases report
We prepared reports for the Domain Adaption section:

| Target   | Source   | F1-score   | PR-score      | W&B |
| -------- | -------- | ---------- | ------------- | --- |
| DC-1     | NJ-101   | 0.67@0.256 | 0.741 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/7845cv77) |
| DC-1     | JY-381-2 | 0.23@0.093 | 0.285 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/1bl3g497) |
| DC-1     | LC-101   | 0.58@0.657 | 0.754 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/2qkzqiwr) |
| DC-1     | XH-1     | 0.63@0.142 | 0.76 mAP@0.5  | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/3v637rsz) |
| JY-381-4 | JY-381-2 | 0.69@0.526 | 0.777 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/lp8dl3d4) |
| JY-381-4 | LC-201   | 0.68@0.446 | 0.891 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/8ekxmk79) |
| JY-381-4 | NJ-201   | 0.69@0.494 | 0.726 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/1ngcalqv) |
| NJ-101   | XH-1     | 0.59@0.122 | 0.698 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/31r7hvb6) |
| NJ-101   | LC-101   | 0.52@0.596 | 0.644 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/3rmd0doh) |
| XH-4     | LC-201   | 0.38@0.618 | 0.408 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/1iretvnb) |
| XH-4     | NJ-201   | 0.39@0.231 | 0.423 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/cjb0e2l6) |
| XH-4     | XH-3     | 0.39@0.415 | 0.396 mAP@0.5 | [link](https://wandb.ai/erikvalle/RTLAOI-DD-YOLOv5/runs/1u5yyl33) |

## Implementation details
Our experiments were performed on a PC with an Intel(R) Core(TM) i5-10400F 2.90GHz CPU and an NVIDIA RTX 3060 GPU. Regarding neural network models, we utilized [YOLOv5](https://github.com/ultralytics/yolov5) because it offers model scaling, is easy to implement and modify. Its architecture loads pre-trained weights in COCO from their respective official repositories. This model was trained with a batch size of 11 on a single GPU for 50 epochs. 

## Other results
Please refer to the paper.