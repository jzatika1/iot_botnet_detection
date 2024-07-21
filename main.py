import argparse
import yaml
import traceback
from utils.logger import Logger
from data.loader import get_data_loader
from features import get_feature_engineer
from models.ml.xgboost_model import create_xgboost_model, train_xgboost_model
from models.dl.lstm_cnn_model import create_lstm_cnn_model, train_lstm_cnn_model
import os
import numpy as np
import json
from evaluation.metrics import evaluate_model
import pandas as pd
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

MODEL_TYPE_TO_DIR = {
    'xgboost': 'ml',
    'lstm_cnn': 'dl'
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="IoT Botnet Detection System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--classify", action="store_true", help="Classify using trained models")
    parser.add_argument("--task", choices=['xgboost', 'lstm_cnn', 'both'], default='both', 
                        help="Select the task to run: xgboost, lstm_cnn, or both")
    return parser.parse_args()

def preprocess_data(config, logger, task, data_loader):
    logger.info("Preprocessing data")
    feature_engineers = get_feature_engineer(task, config, logger, data_loader)
    combined_features, combined_labels = data_loader.load_data(mode='preprocess')
    
    for model_type in (['xgboost', 'lstm_cnn'] if task == 'both' else [task]):
        engineer = feature_engineers[model_type] if task == 'both' else feature_engineers
        dir_type = MODEL_TYPE_TO_DIR[model_type]
        
        processed_features, processed_labels = engineer.engineer_features(combined_features, combined_labels, "combined_dataset")
        if processed_features is not None and processed_labels is not None:
            engineer.save_processed_data(processed_features, processed_labels, config['data']['processed_dir'][dir_type], dir_type)
    
    logger.info("Preprocessing completed.")
    
def process_model(model_type, config, logger, features, labels, num_classes, is_training=True):
    if model_type == 'xgboost':
        model = create_xgboost_model(config, logger, num_classes)
        model_path = os.path.join(config['output']['model_dir'], 'xgboost_model.joblib')
        if is_training:
            trained_model = train_xgboost_model(model, features, labels, num_classes)
            trained_model.save_model(model_path)
        else:
            model.load_model(model_path)
    else:  # lstm_cnn
        input_shape = (features.shape[1], 1)
        model = create_lstm_cnn_model(config, input_shape=input_shape, num_classes=num_classes, logger=logger)
        model_path = os.path.join(config['output']['model_dir'], 'lstm_cnn_model.keras')
        if is_training:
            trained_model = train_lstm_cnn_model(model, features, labels)
            trained_model.save_model(model_path)
        else:
            model.load_model(model_path)
    
    return model, model_path

def train_models(config, logger, task, data_loader):
    logger.info("Training models")
    feature_engineers = get_feature_engineer(task, config, logger, data_loader)
    
    for model_type in (['xgboost', 'lstm_cnn'] if task == 'both' else [task]):
        logger.info(f"Training {model_type.upper()} model")
        engineer = feature_engineers[model_type] if task == 'both' else feature_engineers
        dir_type = MODEL_TYPE_TO_DIR[model_type]
        features, labels, metadata = engineer.load_processed_data(config['data']['processed_dir'][dir_type], dir_type)
        num_classes = metadata['num_classes']
        
        model, model_path = process_model(model_type, config, logger, features, labels, num_classes, is_training=True)
        logger.info(f"{model_type.upper()} model trained and saved to {model_path}")

def classify_data(config, logger, task, data_loader):
    logger.info("Classifying input data")
    feature_engineers = get_feature_engineer(task, config, logger, data_loader)
    
    input_data = data_loader.load_data(mode='classify')
    if input_data is None:
        logger.error("No input data to classify")
        return
    
    results = {}
    
    for model_type in (['xgboost', 'lstm_cnn'] if task == 'both' else [task]):
        engineer = feature_engineers[model_type] if task == 'both' else feature_engineers
        dir_type = MODEL_TYPE_TO_DIR[model_type]
        
        model_results = []
        for features, labels in input_data:
            processed_features, processed_labels = engineer.engineer_features(features, labels, "input_data")
            
            _, metadata = engineer.load_processed_data(config['data']['processed_dir'][dir_type], dir_type)[1:]
            num_classes = metadata['num_classes']
            
            model, _ = process_model(model_type, config, logger, processed_features, processed_labels, num_classes, is_training=False)
            
            performance = classify_dataset(model_type, model, processed_features, processed_labels, config, logger)
            model_results.append(performance)
        
        results[model_type.upper()] = model_results
    
    results_path = os.path.join(config['output']['results_dir'], 'input_classification_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Classification results saved to {results_path}")


def classify_dataset(model_type, model, features, true_labels, config, logger):
    if features is None:
        logger.error("Features are None. Cannot proceed with classification.")
        return None

    logger.info(f"Classifying dataset with {model_type} model")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"True labels shape: {true_labels.shape}")

    try:
        if model_type == 'lstm_cnn':
            features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
            predictions = model.predict(features_reshaped)
        else:
            predictions = model.predict(features)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Convert probabilities to class predictions
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # Get unique classes in true labels
        valid_classes = np.unique(true_labels)
        
        # Map predictions to valid classes
        predictions_mapped = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            if pred in valid_classes:
                predictions_mapped[i] = pred
            else:
                # Assign to the nearest valid class
                predictions_mapped[i] = valid_classes[np.argmin(np.abs(valid_classes - pred))]
        
        logger.info(f"Unique values in y_true: {np.unique(true_labels)}")
        logger.info(f"Unique values in y_pred (before mapping): {np.unique(predictions)}")
        logger.info(f"Unique values in y_pred (after mapping): {np.unique(predictions_mapped)}")
        
        performance = evaluate_model(true_labels, predictions_mapped, model_type.upper(), config, logger)
        return performance
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Features causing error:\n{features}")
        raise

def main():
    logger = Logger.setup('main')
    
    try:
        args = parse_arguments()
        
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        Logger.set_global_debug_mode(args.debug)
        
        logger.info("Starting IoT Botnet Detection System")
        
        data_loader = get_data_loader(config, logger=logger)
        
        if args.preprocess:
            preprocess_data(config, logger, args.task, data_loader)
        if args.train:
            train_models(config, logger, args.task, data_loader)
        if args.classify:
            classify_data(config, logger, args.task, data_loader)
        
        logger.info("IoT Botnet Detection System completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("IoT Botnet Detection System execution finished")

if __name__ == "__main__":
    main()