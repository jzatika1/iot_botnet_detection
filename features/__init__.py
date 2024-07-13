from .feature_engineer import FeatureEngineer
from .ml_feature_engineer import MLFeatureEngineer
from .dl_feature_engineer import DLFeatureEngineer

def get_feature_engineer(task, config, logger=None):
    if task == 'xgboost':
        return MLFeatureEngineer(config, logger)
    elif task == 'lstm_cnn':
        return DLFeatureEngineer(config, logger)
    elif task == 'both':
        return {
            'xgboost': MLFeatureEngineer(config, logger),
            'lstm_cnn': DLFeatureEngineer(config, logger)
        }
    else:
        raise ValueError(f"Invalid task: {task}. Choose 'xgboost', 'lstm_cnn', or 'both'.")