from features import get_feature_engineer
from models.ml.xgboost_model import create_xgboost_model, train_xgboost_model
from models.dl.lstm_cnn_model import create_lstm_cnn_model, train_lstm_cnn_model
import os
import numpy as np

def train_models(config, logger, task):
    logger.info("Training models")
    feature_engineers = get_feature_engineer(task, config, logger)
    
    if task in ['xgboost', 'both']:
        logger.info("Training XGBoost model")
        engineer = feature_engineers if task == 'xgboost' else feature_engineers['xgboost']
        features, labels, metadata = engineer.load_processed_data(config['data']['processed_dir']['ml'], 'ml')
        num_classes = metadata['num_classes']
        train_model('xgboost', features, labels, num_classes, config, logger)
    
    if task in ['lstm_cnn', 'both']:
        logger.info("Training LSTM-CNN model")
        engineer = feature_engineers if task == 'lstm_cnn' else feature_engineers['lstm_cnn']
        features, labels, metadata = engineer.load_processed_data(config['data']['processed_dir']['dl'], 'dl')
        num_classes = metadata['num_classes']
        train_model('lstm_cnn', features, labels, num_classes, config, logger)

def train_model(model_type, features, labels, num_classes, config, logger):
    if model_type == 'xgboost':
        model = create_xgboost_model(config, logger, num_classes)
        trained_model = train_xgboost_model(model, features, labels, num_classes)
        model_path = os.path.join(config['output']['model_dir'], 'xgboost_model.joblib')
    else:  # lstm_cnn
        model = create_lstm_cnn_model(config, input_shape=(features.shape[1], 1), num_classes=num_classes, logger=logger)
        trained_model = train_lstm_cnn_model(model, features, labels)
        model_path = os.path.join(config['output']['model_dir'], 'lstm_cnn_model.keras')
    
    trained_model.save_model(model_path)
    logger.info(f"{model_type.upper()} model saved")