import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from utils.logger import Logger

class ModelEvaluator:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or Logger.setup('ModelEvaluator')
        self.metrics = self.config['evaluation']['metrics']

    def evaluate_model(self, y_true, y_pred, model_name, y_pred_proba=None):
        """
        Evaluate the model performance using metrics specified in the config.

        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            model_name (str): Name of the model being evaluated
            y_pred_proba (np.array, optional): Predicted probabilities for AUC-ROC calculation

        Returns:
            dict: A dictionary containing the evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name} model")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        unique_labels = np.unique(y_true)
        num_classes = len(unique_labels)
        is_binary = num_classes == 2

        results = {}

        for metric in self.metrics:
            if metric == 'accuracy':
                results['accuracy'] = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                results['precision'] = precision_score(y_true, y_pred, average='macro')
            elif metric == 'recall':
                results['recall'] = recall_score(y_true, y_pred, average='macro')
            elif metric == 'f1_score':
                precision = precision_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
                results['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            elif metric == 'confusion_matrix':
                results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            elif metric == 'auc_roc' and y_pred_proba is not None:
                results['auc_roc'] = self.calculate_auc_roc(y_true, y_pred_proba)

        for metric, value in results.items():
            if metric != 'confusion_matrix':
                self.logger.info(f"{metric.capitalize()}: {value:.4f}")
            else:
                self.logger.info(f"Confusion Matrix:\n{value}")

        if not is_binary and 'per_class_metrics' in self.metrics:
            self.log_per_class_metrics(y_true, y_pred, unique_labels)

        return results

    def log_per_class_metrics(self, y_true, y_pred, labels):
        """
        Log per-class precision, recall, and F1-score for multi-class problems.

        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            labels (np.array): Unique labels in the dataset
        """
        precision = precision_score(y_true, y_pred, average=None, labels=labels)
        recall = recall_score(y_true, y_pred, average=None, labels=labels)
        f1_scores = []

        for p, r in zip(precision, recall):
            f1 = 2 * (p * r) / (p + r) if (p + r) != 0 else 0
            f1_scores.append(f1)

        for label, p, r, f in zip(labels, precision, recall, f1_scores):
            self.logger.info(f"Class {label} - Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f:.4f}")

    def calculate_auc_roc(self, y_true, y_pred_proba):
        """
        Calculate the Area Under the ROC Curve (AUC-ROC) for binary and multi-class problems.

        Args:
            y_true (np.array): True labels
            y_pred_proba (np.array): Predicted probabilities

        Returns:
            float or dict: AUC-ROC score (for binary) or dictionary of AUC-ROC scores (for multi-class)
        """
        unique_labels = np.unique(y_true)
        num_classes = len(unique_labels)
        is_binary = num_classes == 2

        if is_binary:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            self.logger.info(f"AUC-ROC: {auc_roc:.4f}")
            return auc_roc
        else:
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            
            auc_roc = {}
            for i, class_label in enumerate(unique_labels):
                auc_roc[class_label] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                self.logger.info(f"AUC-ROC for class {class_label}: {auc_roc[class_label]:.4f}")
            
            macro_auc_roc = np.mean(list(auc_roc.values()))
            self.logger.info(f"Macro Average AUC-ROC: {macro_auc_roc:.4f}")
            
            auc_roc['macro_avg'] = macro_auc_roc
            return auc_roc

def evaluate_model(y_true, y_pred, model_name, config, logger=None, y_pred_proba=None):
    """
    Wrapper function to evaluate a model's performance.

    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        model_name (str): Name of the model being evaluated
        config (dict): Configuration dictionary
        logger (Logger, optional): Logger instance
        y_pred_proba (np.array, optional): Predicted probabilities for AUC-ROC calculation

    Returns:
        dict: A dictionary containing the evaluation metrics
    """
    evaluator = ModelEvaluator(config, logger)
    return evaluator.evaluate_model(y_true, y_pred, model_name, y_pred_proba)
