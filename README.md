# IoT Botnet Detection

IoT Botnet Detection System using Machine Learning (ML) and Deep Learning (DL) models.

## Performance Metrics

### Machine Learning (ML) Model:
- **Accuracy**: 0.9995
- **Precision**: 0.8862
- **Recall**: 0.8024
- **F1 Score**: 0.8422

### Deep Learning (DL) Model:
- **Accuracy**: 0.9995
- **Precision**: 0.7699
- **Recall**: 0.7325
- **F1 Score**: 0.7508

## Requirements

- Conda (Anaconda or Miniconda)

## Installation

1. Create and activate the Conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate iot-botnet-detection
    ```

## Usage

Run the main script with the desired options. Use the `-h` or `--help` flag to see the available options.

```sh
python main.py -h
```

Preprocess XGBoost

```sh
python main.py --preprocess --task xgboost
```

Preprocess LSTM CNN

```sh
python main.py --preprocess --task lstm_cnn
```

Train XGBoost

```sh
python main.py --train --task xgboost
```

Train LSTM CNN

```sh
python main.py --train --task lstm_cnn
```

Classify With XGBoost

```sh
python main.py --classify --task xgboost
```

Classify With LSTM CNN

```sh
python main.py --classify --task lstm_cnn
```