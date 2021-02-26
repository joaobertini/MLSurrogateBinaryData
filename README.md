# Machile Learning Surrogate models
Codes from paper "A comparison of machine learning surrogate models for net present value prediction from well placement high-dimensional binary data"

## Dependencies

```bash
pip3 install numpy pandas matplotlib scikit-learn
```

## Running

```bash
python3 run.py DATASET_NAME MODEL_NAME REDUCER_NAME REDUCER_DIMENSION
```

Examples:
```bash
python3 run.py dataUNISIM1 GTB PCA 5
```

```bash
python3 run.py dataUNISIM1 KRR NONE 0
```

