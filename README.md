# IoT Botnet Detection

IoT Botnet Detection System using Machine Learning (ML) and Deep Learning (DL) models.

## Classification Considerations & Limitations

When training the model on a dataset or datasets, it is crucial to ensure that the features and labels of the input data match those used during training. This is important for accurate classification results.

Using Principal Component Analysis (PCA) is a great approach for dimensionality reduction and feature selection. However, further care must be taken to ensure that the features selected by PCA align with the features that will be available in real-time for classification. This involves:

1. **Feature Matching**: Ensuring that the input features used for real-time classification are the same as those used to train the model. Any discrepancy in the feature set can lead to incorrect classifications.

2. **Label Matching**: Ensuring that the labels used for classification are consistent with those used during model training. Inconsistent labels can result in misclassification or model errors.

3. **Real-Time Access**: Verifying that the features selected by PCA or any other feature selection method are readily available and can be accessed in real-time. This ensures that the model can be effectively deployed in a live environment.

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