import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils.logger import Logger
from evaluation.metrics import evaluate_model

class LSTMCNNModel:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or Logger.setup('LSTMCNNModel')
        self.model = None
        self.dl_config = config['models']['dl']['lstm_cnn']
        self.data_split_config = config['training']
        self.gpu_acceleration = self.config['models'].get('gpu_acceleration', True)
        self.num_classes = None

        if self.gpu_acceleration:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                try:
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                    self.logger.info(f"Using GPU: {physical_devices[0].name}")
                except RuntimeError as e:
                    self.logger.error(f"Failed to set GPU memory growth: {e}")
                    self.gpu_acceleration = False
            else:
                self.logger.info("No GPU found, falling back to CPU.")
                self.gpu_acceleration = False

    def create_model(self, input_shape, num_classes):
        self.num_classes = num_classes
        device = '/gpu:0' if self.gpu_acceleration else '/cpu:0'
        with tf.device(device):
            self.model = Sequential([
                LSTM(units=self.dl_config['lstm_units'], return_sequences=True, input_shape=input_shape),
                LSTM(units=self.dl_config['lstm_units'], return_sequences=True),
                Conv1D(filters=self.dl_config['cnn_filters'], kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(units=self.dl_config['dense_units'], activation='relu'),
                Dropout(rate=self.dl_config['dropout_rate']),
                Dense(units=num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=self.dl_config['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        self.logger.info(f"Created LSTM-CNN model with input shape: {input_shape}, num_classes: {num_classes}")
        self.model.summary(print_fn=self.logger.info)

    def train(self, X, y):
        self.logger.info("Starting LSTM+CNN model training")
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.data_split_config['test_size'],
            random_state=self.data_split_config['random_state']
        )

        device = '/gpu:0' if self.gpu_acceleration else '/cpu:0'
        with tf.device(device):
            history = self.model.fit(
                X_train, y_train,
                epochs=self.dl_config['epochs'],
                batch_size=self.dl_config['batch_size'],
                validation_data=(X_val, y_val),
                verbose=1
            )
    
        y_pred = self.predict(X_val)
        y_pred_proba = self.model.predict(X_val)

        # Evaluate using ModelEvaluator
        evaluate_model(y_val, y_pred, "LSTM-CNN", self.config, self.logger, y_pred_proba)
    
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Reshape input to 3D if necessary
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return np.argmax(self.model.predict(X), axis=-1)

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Ensure the file has the .keras extension
        if not filepath.endswith('.keras'):
            filepath = os.path.splitext(filepath)[0] + '.keras'
        
        self.model.save(filepath, save_format='keras')
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        # Ensure the file has the .keras extension
        if not filepath.endswith('.keras'):
            filepath = os.path.splitext(filepath)[0] + '.keras'
        
        self.model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        
        # Infer num_classes from the loaded model
        self.num_classes = self.model.layers[-1].output_shape[-1]
        self.logger.info(f"Loaded model has {self.num_classes} classes")

def create_lstm_cnn_model(config, input_shape, num_classes, logger=None):
    model = LSTMCNNModel(config, logger)
    model.create_model(input_shape=input_shape, num_classes=num_classes)
    return model

def train_lstm_cnn_model(model, X, y):
    # Ensure X is reshaped to 3D (samples, timesteps, features)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return model.train(X, y)