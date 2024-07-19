from features import get_feature_engineer
from models.ml.xgboost_model import create_xgboost_model
from models.dl.lstm_cnn_model import create_lstm_cnn_model
from evaluation.metrics import evaluate_model
from data.loader import get_data_loader
import os
import numpy as np
import pandas as pd
import json

def classify_data(config, logger, task):
    logger.info("Classifying input data")
    feature_engineers = get_feature_engineer(task, config, logger)
    data_loader = get_data_loader(config, logger)
    
    # Load input data
    input_data = data_loader.load_input_data()
    if input_data is None:
        logger.error("No input data to classify")
        return
    
    # Extract features and labels
    features, labels = data_loader.extract_features_and_labels(input_data, "input_data")
    if features is None or labels is None:
        logger.error("Failed to extract features and labels from input data")
        return
    
    logger.info(f"Extracted features shape: {features.shape}, labels shape: {labels.shape}")
    
    results = {}
    
    if task in ['xgboost', 'both']:
        engineer = feature_engineers if task == 'xgboost' else feature_engineers['xgboost']
        processed_features, processed_labels = engineer.engineer_features(features, labels, "input_data")
        results["XGBoost"] = classify_dataset('xgboost', processed_features, processed_labels, config, logger)
    
    if task in ['lstm_cnn', 'both']:
        engineer = feature_engineers if task == 'lstm_cnn' else feature_engineers['lstm_cnn']
        processed_features, processed_labels = engineer.engineer_features(features, labels, "input_data")
        results["LSTM-CNN"] = classify_dataset('lstm_cnn', processed_features, processed_labels, config, logger)
    
    results_path = os.path.join(config['output']['results_dir'], 'input_classification_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Classification results saved to {results_path}")

def classify_dataset(model_type, features, true_labels, config, logger):
    if features is None:
        logger.error("Features are None. Cannot proceed with classification.")
        return None

    logger.info(f"Classifying dataset with {model_type} model")
    logger.info(f"Features type: {type(features)}")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"True labels shape: {true_labels.shape}")
    
    if isinstance(features, pd.DataFrame):
        logger.info("Features data types:")
        for column, dtype in features.dtypes.items():
            logger.info(f"  {column}: {dtype}")
        logger.info(f"Sample of features:\n{features.head()}")
    elif isinstance(features, np.ndarray):
        logger.info(f"Features data type: {features.dtype}")
        logger.info(f"Sample of features:\n{features[:5]}")
    else:
        logger.warning(f"Unexpected features type: {type(features)}")

    if model_type == 'xgboost':
        model_path = os.path.join(config['output']['model_dir'], 'xgboost_model.joblib')
        model = create_xgboost_model(config, logger)
        model.load_model(model_path)
        logger.info("XGBoost model loaded successfully")
    else:  # lstm_cnn
        model_path = os.path.join(config['output']['model_dir'], 'lstm_cnn_model.keras')
        
        metadata_path = os.path.join(config['data']['processed_dir']['dl'], 'dl_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_classes = metadata['num_classes']
        
        input_shape = (features.shape[1], 1)
        model = create_lstm_cnn_model(config, input_shape=input_shape, num_classes=num_classes, logger=logger)
        model.load_model(model_path)
        logger.info(f"LSTM-CNN model loaded successfully. Input shape: {input_shape}, num_classes: {num_classes}")

    try:
        logger.info("Starting prediction")
        if model_type == 'lstm_cnn':
            features_array = features.values if isinstance(features, pd.DataFrame) else features
            features_reshaped = features_array.reshape(features_array.shape[0], features_array.shape[1], 1)
            predictions = model.predict(features_reshaped)
        else:
            predictions = model.predict(features)
        
        logger.info(f"Raw predictions shape: {predictions.shape}")
        logger.info(f"Sample of raw predictions: {predictions[:5]}")
        
        # Convert continuous outputs to discrete class predictions
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            logger.info("Converting continuous outputs to discrete class predictions")
            predictions = np.argmax(predictions, axis=1)
        
        logger.info(f"Final predictions shape: {predictions.shape}")
        logger.info(f"Sample of final predictions: {predictions[:5]}")
        logger.info(f"Unique predicted classes: {np.unique(predictions)}")

        if true_labels is not None:
            logger.info(f"True labels shape: {true_labels.shape}")
            logger.info(f"Sample of true labels: {true_labels[:5]}")
            logger.info(f"Unique true classes: {np.unique(true_labels)}")
            
            if predictions.shape != true_labels.shape:
                logger.error(f"Shape mismatch: predictions {predictions.shape}, true_labels {true_labels.shape}")
                return None
            
            performance = evaluate_model(true_labels, predictions, model_type.upper(), config, logger)
            return performance
        else:
            return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Features causing error:\n{features}")
        raise