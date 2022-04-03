# IDL Homwork2 part2

(11-785 Introduction to Deep Learning)

Face Classification & Verification using CNN


## Models

I trained two models for these tasks.

- `` MLP `` 
- ``mobilenetv2`` # Number of Params: 11189128


## Dataset

The training/validation datasets were provided from the instructors.

# How to run the implemendted codes

## 1. Install Requirements

```
pip install -r requirements.txt
```

## 2. File System

```
    Classification
    │ 
    ├── model
    │   └── mobilenetv2.py
    ├── train
    │   ├── train_mobilenetv2.py
    │   └── addi_train.py
    └── test
        └── test.py

    Verification
    └── ver.py
```

## 3. How to run

### Classification
```bash
$ RUN train_mobilenetv2.py # training and evaluation code
$ RUN test.py # inference code
```

- To additional train
```
RUN addi_train.py
```

### Verification
```
RUN ver.py
```