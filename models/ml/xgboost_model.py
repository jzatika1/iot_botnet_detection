import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils.logger import Logger
from evaluation.metrics import evaluate_model
import joblib
import os

class XGBoostModel:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or Logger.setup('XGBoostModel')
        self.model = None
        self.xgb_config = config['models']['ml']['xgboost']
        self.data_split_config = config['training']
        self.gpu_acceleration = self.config['models'].get('gpu_acceleration', False)

    def create_model(self, num_classes):
        params = self.xgb_config.copy()
        early_stopping_rounds = params.pop('early_stopping_rounds', 10)
        
        if self.gpu_acceleration:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            self.logger.info("GPU acceleration is enabled for XGBoost.")
        else:
            params['tree_method'] = 'hist'
            self.logger.info("Using CPU for XGBoost.")
        
        params['num_class'] = num_classes
        self.model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        self.logger.info(f"Created XGBoost model with parameters: {self.model.get_params()}")
        return self.model

    def train(self, X, y):
        self.logger.info("Starting XGBoost model training")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.data_split_config['test_size'],
            random_state=self.data_split_config['random_state']
        )
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)
        evaluate_model(y_val, y_pred, "XGBoost", self.config, self.logger, y_pred_proba)
        
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        self.logger.info(f"Predicting with input type: {type(X)}")
        self.logger.info(f"Input shape: {X.shape}")
        
        try:
            if self.gpu_acceleration:
                self.logger.info("Using GPU for prediction")
                return self.model.predict(X, output_margin=True)
            else:
                self.logger.info("Using CPU for prediction")
                return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        self.logger.info(f"Loaded model parameters: {self.model.get_params()}")

def create_xgboost_model(config, logger=None, num_classes=None):
    model = XGBoostModel(config, logger)
    if num_classes is not None:
        model.create_model(num_classes)
    return model

def train_xgboost_model(model, X, y, num_classes):
    if model.model is None:
        model.create_model(num_classes)
    return model.train(X, y)