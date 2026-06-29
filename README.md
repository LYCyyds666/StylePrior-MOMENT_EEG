# `SageStream` Artifact

**This repository is the official Pytorch implementation of "SageStream: Bridging the Prior Gap via Time-Series Foundation Models for Cross-Subject Medical Time-Series Classification".**

## 1.0 Instructions

```
├── cache/                    
├── datasets/                 // Datasets
│   ├── APAVA.zip            
├── MoE_moment/              // Framework
│   ├── momentfm/            
│   │   ├── models/          // Model implementations
├── MOMENT-1-small/          // Pre-trained MOMENT model
├── preprocessing.py         // Preprocessing
├── two_stage_training.py    // Training pipeline
└── utils.py                 // Utility
```

Among them, the SageStream-related implementation is in the directory:
* `MoE_moment/momentfm/models/SS_MOMENT.py`
* `MoE_moment/momentfm/models/layers/SA_MoE.py`
* `MoE_moment/momentfm/models/layers/SA_MoE_components.py`

## 2.0 Preparation
**Dataset Preparation**

```
$ cd datasets
$ unzip APAVA.zip
```


**Pre-trained Model Preparation**
Download the files of MOMENT from Hugging Face and place them in `./MOMENT-1-small`


## 3.0 Run SageStream

> python two_stage_training.py