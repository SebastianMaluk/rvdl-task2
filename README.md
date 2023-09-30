# How to run

## Install dependencies
```bash
pip install -r requirements.txt
```

## Download dataset
Uncompress the [dataset](https://tinyurl.com/v2dj69y9) in the root folder of the project.

## Run split script
```bash
python tfds_sketchy/split.py
```

## Build dataset
```bash
cd tensorflow_datasets/sketchy
tfds build
```

## Run notebok main.ipynb
