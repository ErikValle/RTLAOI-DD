# Relational based Transfer Learning for Automatic Optical Inspection based on domain discrepancy

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
This Jupyter notebook contains a demonstration of how we calculated the Earth Mover’s Distance for domain discrepancy estimation from stratch.