import pandas as pd
from tqdm import tqdm
from features import get_feature_engineer

def preprocess_data(config, logger, task, data_loader):
    logger.info("Preprocessing data")
    feature_engineers = get_feature_engineer(task, config, logger)
    
    combined_features, combined_labels = data_loader.load_and_combine_datasets()
    
    logger.info(f"Combined dataset shape: Features {combined_features.shape}, Labels {combined_labels.shape}")
    
    if task in ['xgboost', 'both']:
        engineer = feature_engineers if task == 'xgboost' else feature_engineers['xgboost']
        processed_features_ml, processed_labels_ml = engineer.engineer_features(combined_features, combined_labels, "combined_dataset")
        if processed_features_ml is not None and processed_labels_ml is not None:
            engineer.save_processed_data(processed_features_ml, processed_labels_ml, config['data']['processed_dir']['ml'], 'ml')
    
    if task in ['lstm_cnn', 'both']:
        engineer = feature_engineers if task == 'lstm_cnn' else feature_engineers['lstm_cnn']
        processed_features_dl, processed_labels_dl = engineer.engineer_features(combined_features, combined_labels, "combined_dataset")
        if processed_features_dl is not None and processed_labels_dl is not None:
            engineer.save_processed_data(processed_features_dl, processed_labels_dl, config['data']['processed_dir']['dl'], 'dl')

    logger.info("Preprocessing completed.")