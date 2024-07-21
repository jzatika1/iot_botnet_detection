import os
import re
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils.logger import Logger
import yaml
import numpy as np
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

class DataLoader:
    def __init__(self, config, logger=None):
        self.config = config
        self.raw_dir = self.config['data']['raw_dir']
        self.input_dir = self.config['data']['input_dir']
        self.batch_size = self.config['data']['batch_size']
        self.logger = logger or Logger.setup('DataLoader')

    def load_data(self, mode='preprocess'):
        if mode == 'preprocess':
            return self.load_and_combine_datasets()
        elif mode == 'classify':
            return self.load_input_data()
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'preprocess' or 'classify'.")

    def load_input_data(self):
        self.logger.info(f"Loading input data from {self.input_dir}")
        
        all_data = []
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            data = self.load_file(file_path)
            if data is not None:
                features, labels = self.extract_features_and_labels(data, "input_data")
                features_clean, labels_clean = self.clean_and_filter_data(features, labels)
                all_data.append((features_clean, labels_clean))
        
        if all_data:
            self.logger.info(f"Loaded {len(all_data)} input files")
            return all_data
        else:
            self.logger.warning("No input data loaded")
            return None

    def load_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                # Check if this is an UNSW-NB15 file
                if 'UNSW-NB15' in file_path:
                    column_names = [
                        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
                        'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
                        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                        'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
                        'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                        'attack_cat', 'Label'
                    ]
                    try:
                        data = pd.read_csv(file_path, names=column_names, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.logger.warning(f"UTF-8 decoding failed for {file_path}. Trying ISO-8859-1.")
                        data = pd.read_csv(file_path, names=column_names, encoding='ISO-8859-1')
                else:
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.logger.warning(f"UTF-8 decoding failed for {file_path}. Trying ISO-8859-1.")
                        data = pd.read_csv(file_path, encoding='ISO-8859-1')
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_path}")
                return None
            
            # Strip whitespace from column names
            data.columns = data.columns.str.strip()
            
            return data
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def process_dataset(self, dataset_name):
        dataset_config = self.config['data']['datasets'][dataset_name]
        if not dataset_config['enabled']:
            self.logger.info(f"Skipping disabled dataset: {dataset_name}")
            return None, None
    
        dataset_path = os.path.join(self.raw_dir, dataset_name)
        self.logger.info(f"Processing dataset: {dataset_name}")
    
        file_paths = [os.path.join(dataset_path, f) for f in dataset_config['files']]
        self.logger.info(f"Found {len(file_paths)} files in {dataset_name}")
    
        all_data = []
        file_data_lengths = {}
    
        with Pool(processes=cpu_count()) as pool:
            self.logger.info(f"Using a pool of {cpu_count()} processes")
            for i in range(0, len(file_paths), self.batch_size):
                batch_files = file_paths[i:i + self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch_files)} files")
                batch_data = pool.map(self.load_file, batch_files)
                for file, data in zip(batch_files, batch_data):
                    if data is not None:
                        all_data.append(data)
                        file_data_lengths[file] = len(data)
                        Logger.debug(self.logger, f"Processed file: {file}, loaded {len(data)} records")
                self.logger.info(f"Batch {i//self.batch_size + 1} processed. Total records so far: {sum(file_data_lengths.values())}")
    
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Processed {dataset_name} dataset: {len(file_paths)} files, {len(combined_data)} total records")
            return combined_data, file_data_lengths
        else:
            self.logger.warning(f"No data loaded for dataset: {dataset_name}")
            return None, None

    def load_datasets(self):
        datasets = {}
        for dataset_name in self.config['data']['datasets']:
            if self.config['data']['datasets'][dataset_name]['enabled']:
                data, file_lengths = self.process_dataset(dataset_name)
                if data is not None:
                    datasets[dataset_name] = (data, file_lengths)
        return datasets

    def clean_and_filter_data(self, features, labels, threshold=1e15):
        initial_rows = len(features)
        numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
        
        # Create masks for different types of problematic data
        nan_mask = features.isna().any(axis=1)
        inf_mask = features[numeric_columns].isin([np.inf, -np.inf]).any(axis=1)
        extreme_mask = (features[numeric_columns].abs() > threshold).any(axis=1)
        
        # Combine all masks
        problematic_mask = nan_mask | inf_mask | extreme_mask
        
        # Remove problematic rows
        features_clean = features[~problematic_mask]
        labels_clean = labels[~problematic_mask]
        
        # Count removed rows
        nan_removed = nan_mask.sum()
        inf_removed = inf_mask.sum()
        extreme_removed = extreme_mask.sum()
        total_removed = problematic_mask.sum()
        
        self.logger.info(f"Data cleaning: Removed {total_removed} rows ({nan_removed} NaN, {inf_removed} infinite, {extreme_removed} extreme). Remaining rows: {len(features_clean)}")
        
        return features_clean, labels_clean

    def load_and_combine_datasets(self):
        datasets = self.load_datasets()
        
        all_features = []
        all_labels = []
        
        for dataset_name, (data, _) in tqdm(datasets.items(), desc="Loading and Cleaning Datasets"):
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            features, labels = self.extract_features_and_labels(data, dataset_name)
            if features is None or labels is None:
                self.logger.info(f"Skipping dataset: {dataset_name}")
                continue
            
            # Clean and filter the dataset
            features_clean, labels_clean = self.clean_and_filter_data(features, labels)
            
            all_features.append(features_clean)
            all_labels.append(labels_clean)
        
        combined_features = pd.concat(all_features, axis=0, ignore_index=True)
        combined_labels = pd.concat(all_labels, axis=0, ignore_index=True)
        
        self.logger.info(f"Combined dataset shape: Features {combined_features.shape}, Labels {combined_labels.shape}")
        
        return combined_features, combined_labels

    def extract_features_and_labels(self, data, dataset_name):
        possible_label_columns = ['Label', 'label', 'type', 'Type', 'attack_cat']
        
        # Identify the label column
        label_column = next((col for col in possible_label_columns if col in data.columns), None)
        
        if label_column is None:
            self.logger.error(f"No suitable label column found in dataset '{dataset_name}'. Available columns: {data.columns.tolist()}")
            return None, None
        
        # Extract features and labels
        features = data.drop(columns=[col for col in possible_label_columns if col in data.columns]).copy()
        labels = data[label_column].copy()
        
        self.logger.info(f"Processing dataset: {dataset_name}")
        self.logger.info(f"Label column identified: {label_column}")
        
        # Dataset-specific processing
        if dataset_name.upper() == 'ROUTESMART':
            self.logger.info("ROUTESMART dataset detected. Treating all data as malicious.")
            labels = pd.Series(['malicious'] * len(data), index=data.index)
        elif dataset_name.upper() == 'UNSW-NB15':
            self.logger.info("UNSW-NB15 dataset detected. Using 'attack_cat' as label.")
            nan_mask = labels.isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                self.logger.info(f"Found {nan_count} NaN values in 'attack_cat'. Using 'Label' to categorize them.")
                labels = labels.mask(nan_mask & (data['Label'] == 0), 'benign')
                labels = labels.mask(nan_mask & (data['Label'] == 1), 'malicious')
        
        # Dynamic label renaming
        label_patterns = {
            r'web attack|web_attack': 'web_attack',
            r'brute force|brute_force': 'web_attack',
            r'xss': 'web_attack',
            r'sql injection|sql_injection': 'web_attack',
            r'dos': 'dos',
            r'slowloris': 'dos',
            r'slowhttptest': 'dos',
            r'hulk': 'dos',
            r'goldeneye': 'dos'
        }
        
        def categorize_label(label):
            if not isinstance(label, str):
                label = str(label)
            label = label.lower()
            for pattern, standard_label in label_patterns.items():
                if re.search(pattern, label):
                    return standard_label
            return label.replace(' ', '_')
        
        labels = labels.apply(categorize_label)
        
        # Common label processing for all datasets
        labels = labels.fillna('unlabeled')
        labels = labels.replace('', 'unknown')  # Replace empty strings with 'unknown'
        
        # Log information about labels and features
        label_counts = labels.value_counts(dropna=False)
        total_count = len(labels)
        label_info = "\n".join([f"{label}: {count} ({count/total_count:.2%})" for label, count in label_counts.items()])
        
        self.logger.info(f"Label distribution:\n{label_info}")
        self.logger.info(f"Features shape after extraction: {features.shape}")
        
        return features, labels

def get_data_loader(config, logger=None):
    return DataLoader(config, logger)
