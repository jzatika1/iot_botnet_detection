from .feature_engineer import FeatureEngineer
from .ml_feature_engineer import MLFeatureEngineer
from .dl_feature_engineer import DLFeatureEngineer

def get_feature_engineer(task, config, logger, data_loader):
    if task == 'xgboost':
        return MLFeatureEngineer(config, logger, data_loader)
    elif task == 'lstm_cnn':
        return DLFeatureEngineer(config, logger, data_loader)
    elif task == 'both':
        return {
            'xgboost': MLFeatureEngineer(config, logger, data_loader),
            'lstm_cnn': DLFeatureEngineer(config, logger, data_loader)
        }
    else:
        raise ValueError(f"Invalid task: {task}. Expected 'xgboost', 'lstm_cnn', or 'both'.")