# IDL Homwork3 part2

(11-785 Introduction to Deep Learning)

Utterance to Phoneme Mapping using Sequence Models


## Models

I used LSTM for this task.

- Feature embedding : reduced ResNet
- LSTM : (2048, 1024, 2) with bi-directional


## Dataset

The training/validation datasets were provided from the instructors.

# How to run the implemented codes

## 1. Install Requirements

- check your nvcc version to install torch
```
pip install git+https://github.com/parlance/ctcdecode.git
pip install -r requirements.txt
```

## 2. File Architecture

```
    File Architecture
    │ 
    ├── model
    │   └── model.py
    ├── train
    │   ├── main.py
    │   └── addi_train.py
    ├── test
    │   └── inference.py
    ├── dataset.py # dataloader
    └── utils.py   # to calculate Levenshtein distance

```

## 3. Used technique
- ResNet Block to feature embedding
- Dropout 
- Bi-directional LSTM
- GELU
- AdamW
- CosineAnnealingLR

## 4. How to run

```bash
$ RUN train.sh # training and evaluation code
$ RUN inference.sh # inference code
```

-> To additional train
```
RUN addi_train.py
```
