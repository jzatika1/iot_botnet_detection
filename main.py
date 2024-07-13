import argparse
import yaml
import numpy as np
import pandas as pd
import os
import traceback
from tqdm import tqdm
import json
from utils.logger import Logger
from data.loader import get_data_loader
from features import get_feature_engineer
from models.ml.xgboost_model import create_xgboost_model, train_xgboost_model
from models.dl.lstm_cnn_model import create_lstm_cnn_model, train_lstm_cnn_model
from evaluation.metrics import evaluate_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="IoT Botnet Detection System")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--classify", action="store_true", help="Classify using trained models")
    parser.add_argument("--task", choices=['xgboost', 'lstm_cnn', 'both'], default='both', 
                        help="Select the task to run: xgboost, lstm_cnn, or both")
    return parser

def preprocess_data(config, logger, task, data_loader):
    logger.info("Preprocessing data")
    feature_engineers = get_feature_engineer(task, config, logger)
    
    datasets = data_loader.load_datasets()
    
    all_features_ml = []
    all_labels_ml = []
    all_features_dl = []
    all_labels_dl = []
    
    for dataset_name, (data, _) in tqdm(datasets.items(), desc="Datasets"):
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        target_column = config['preprocessing']['target_column']
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in dataset '{dataset_name}'. Available columns: {data.columns.tolist()}")
            logger.info(f"Skipping dataset: {dataset_name}")
            continue

        features = data.drop(columns=[target_column])
        labels = data[target_column]

        if isinstance(feature_engineers, dict):
            if 'xgboost' in feature_engineers:
                processed_features, processed_labels = process_dataset(features, labels, feature_engineers['xgboost'], f"{dataset_name}_xgboost", config, logger, 'xgboost')
                if processed_features is not None and processed_labels is not None:
                    all_features_ml.append(processed_features)
                    all_labels_ml.append(processed_labels)
            if 'lstm_cnn' in feature_engineers:
                processed_features, processed_labels = process_dataset(features, labels, feature_engineers['lstm_cnn'], f"{dataset_name}_lstm_cnn", config, logger, 'lstm_cnn')
                if processed_features is not None and processed_labels is not None:
                    all_features_dl.append(processed_features)
                    all_labels_dl.append(processed_labels)
        else:
            processed_features, processed_labels = process_dataset(features, labels, feature_engineers, dataset_name, config, logger, task)
            if processed_features is not None and processed_labels is not None:
                if task == 'xgboost':
                    all_features_ml.append(processed_features)
                    all_labels_ml.append(processed_labels)
                elif task == 'lstm_cnn':
                    all_features_dl.append(processed_features)
                    all_labels_dl.append(processed_labels)
    
    # Combine and save all processed data for ML
    if all_features_ml and all_labels_ml:
        combined_features_ml = np.concatenate(all_features_ml, axis=0)
        combined_labels_ml = np.concatenate(all_labels_ml, axis=0)
        
        processed_dir_ml = config['data']['processed_dir']['ml']
        os.makedirs(processed_dir_ml, exist_ok=True)
        
        np.save(os.path.join(processed_dir_ml, 'combined_features.npy'), combined_features_ml)
        np.save(os.path.join(processed_dir_ml, 'combined_labels.npy'), combined_labels_ml)
        
        logger.info(f"Combined ML preprocessed data saved. Features shape: {combined_features_ml.shape}, Labels shape: {combined_labels_ml.shape}")
    else:
        logger.warning("No ML data was preprocessed. Check for errors in individual dataset processing.")

    # Combine and save all processed data for DL
    if all_features_dl and all_labels_dl:
        combined_features_dl = np.concatenate(all_features_dl, axis=0)
        combined_labels_dl = np.concatenate(all_labels_dl, axis=0)
        
        processed_dir_dl = config['data']['processed_dir']['dl']
        os.makedirs(processed_dir_dl, exist_ok=True)
        
        np.save(os.path.join(processed_dir_dl, 'combined_features.npy'), combined_features_dl)
        np.save(os.path.join(processed_dir_dl, 'combined_labels.npy'), combined_labels_dl)
        
        logger.info(f"Combined DL preprocessed data saved. Features shape: {combined_features_dl.shape}, Labels shape: {combined_labels_dl.shape}")
    else:
        logger.warning("No DL data was preprocessed. Check for errors in individual dataset processing.")

def process_dataset(features, labels, feature_engineer, dataset_name, config, logger, task):
    try:
        logger.info(f"Engineering features for {dataset_name}")
        
        # Convert features to numeric if they aren't already
        for col in features.columns:
            if features[col].dtype == 'object':
                logger.info(f"Converting column {col} to numeric")
                features[col] = pd.factorize(features[col])[0]
        
        engineered_features, engineered_labels = feature_engineer.engineer_features(features, labels, dataset_name)
        
        logger.info(f"Preprocessing completed for dataset: {dataset_name}")
        return engineered_features, engineered_labels
    except Exception as e:
        logger.error(f"Error during preprocessing of {dataset_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info(f"Skipping dataset: {dataset_name}")
        return None, None

def train_model(model_type, features, labels, config, logger, dataset_name):
    if model_type == 'xgboost':
        model = create_xgboost_model(config, logger)
        trained_model = train_xgboost_model(model, features, labels)
        model_path = os.path.join(config['output']['model_dir'], f'xgboost_{dataset_name}.joblib')
    else:  # lstm_cnn
        num_classes = len(np.unique(labels))
        model = create_lstm_cnn_model(config, input_shape=(features.shape[1], 1), num_classes=num_classes, logger=logger)
        trained_model = train_lstm_cnn_model(model, features, labels)
        model_path = os.path.join(config['output']['model_dir'], f'lstm_cnn_{dataset_name}.keras')
    
    trained_model.save_model(model_path)
    logger.info(f"{model_type.upper()} model saved for dataset: {dataset_name}")

def train_models(config, logger, task):
    logger.info("Training models")
    
    task_dir_map = {'xgboost': 'ml', 'lstm_cnn': 'dl'}
    
    if task in ['xgboost', 'both']:
        preprocessed_dir = config['data']['processed_dir']['ml']
        logger.info(f"Training XGBoost model")
        features = np.load(os.path.join(preprocessed_dir, 'combined_features.npy'))
        labels = np.load(os.path.join(preprocessed_dir, 'combined_labels.npy'))
        train_model('xgboost', features, labels, config, logger, 'combined')
    
    if task in ['lstm_cnn', 'both']:
        preprocessed_dir = config['data']['processed_dir']['dl']
        logger.info(f"Training LSTM-CNN model")
        features = np.load(os.path.join(preprocessed_dir, 'combined_features.npy'))
        labels = np.load(os.path.join(preprocessed_dir, 'combined_labels.npy'))
        train_model('lstm_cnn', features, labels, config, logger, 'combined')

def classify_dataset(model_type, features, true_labels, config, logger, dataset_name):
    if model_type == 'xgboost':
        model_path = os.path.join(config['output']['model_dir'], f'xgboost_{dataset_name}.joblib')
        model = create_xgboost_model(config, logger)
        model.load_model(model_path)
    else:  # lstm_cnn
        model_path = os.path.join(config['output']['model_dir'], f'lstm_cnn_{dataset_name}.h5')
        num_classes = len(np.unique(true_labels))
        model = create_lstm_cnn_model(config, input_shape=(1, features.shape[1]), num_classes=num_classes, logger=logger)
        model.load_model(model_path)
    
    predictions = model.predict(features)
    performance = evaluate_model(true_labels, predictions, model_type.upper(), logger)
    return performance

def classify_data(config, logger, task, data_loader):
    logger.info("Classifying data")
    test_datasets = data_loader.load_test_datasets()
    
    results = {}
    
    if task in ['xgboost', 'both']:
        preprocessed_dir = config['data']['processed_dir']['ml']
        features = np.load(os.path.join(preprocessed_dir, 'combined_features.npy'))
        true_labels = np.load(os.path.join(preprocessed_dir, 'combined_labels.npy'))
        results["XGBoost"] = classify_dataset('xgboost', features, true_labels, config, logger, 'combined')
    
    if task in ['lstm_cnn', 'both']:
        preprocessed_dir = config['data']['processed_dir']['dl']
        features = np.load(os.path.join(preprocessed_dir, 'combined_features.npy'))
        true_labels = np.load(os.path.join(preprocessed_dir, 'combined_labels.npy'))
        results["LSTM-CNN"] = classify_dataset('lstm_cnn', features, true_labels, config, logger, 'combined')
    
    results_path = os.path.join(config['output']['results_dir'], 'results_combined.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Classification results saved to {results_path}")

def main():
    # Initialize logger first
    logger = Logger.setup('main')
    
    try:
        # Parse arguments
        parser = parse_arguments()
        args = parser.parse_args()
        
        # Load configuration
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        # Set debug mode
        Logger.set_global_debug_mode(args.debug)
        
        logger.info("Starting IoT Botnet Detection System")
        
        # Load data
        data_loader = get_data_loader(config, logger=logger)
        
        if args.preprocess:
            preprocess_data(config, logger, args.task, data_loader)
        if args.train:
            train_models(config, logger, args.task)
        if args.classify:
            classify_data(config, logger, args.task, data_loader)
        
        logger.info("IoT Botnet Detection System completed successfully")
    
    except argparse.ArgumentError as e:
        logger.error(f"Argument parsing error: {str(e)}")
        logger.info("Use --help for usage information")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("IoT Botnet Detection System execution finished")

if __name__ == "__main__":
    main()