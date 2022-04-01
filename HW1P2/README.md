# CMU 11-785 Spring 2022 HW1P2

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
conda env create -f environment.yml
conda activate idl-hw1p2
```


## Model
I made a small MLP model with pytorch environment. There are ReLU activation function, Dropout, Batchh normalization and some Linear layers.

The loss function is Cross entropy loss, and the optimizer is Adam. I also used a scheduler. 

### training & inference directly
```
RUN hw1p2.py
```