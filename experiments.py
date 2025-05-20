import os
import re
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelExperiment:
    def __init__(self, dataset_dir='dataset', embedding_dim=None, openface_windows=5):
        self.dataset_dir = dataset_dir
        self.metadata_path = os.path.join(dataset_dir, 'dataset.jsonl')
        self.features_dir = os.path.join(dataset_dir, 'features')
        self.metadata = self._load_metadata()
        self.embedding_dim = embedding_dim
        self.openface_windows = openface_windows

        # Models: add new models here
        # Using SimpleImputer to handle NaN values before scaling/classification
        self.models = {
            'svm': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('clf', SVC())
            ]),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),  # handles NaNs internally
            'dense': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(
                    hidden_layer_sizes=(64,), 
                    activation='relu',
                    max_iter=200,
                    random_state=42
                ))
            ])
        }

        # Feature loader map for non-audio and non-video features
        self.feature_loaders = {
            'text': lambda vid, iid: self._load_npy('text', f"{vid}_{iid}.npy"),
            'vad': lambda vid, iid: self._load_npy('VAD', f"{vid}_{iid}.npy"),
            'figurative': lambda vid, iid: self._get_figurative_label(vid, iid)
        }

    def _load_metadata(self):
        records = []
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)

    def _load_npy(self, feat_type, filename):
        path = os.path.join(self.features_dir, feat_type, filename)
        try:
            return np.load(path)
        except Exception:
            data = np.fromfile(path, dtype=np.float32)
            if self.embedding_dim and data.size % self.embedding_dim == 0:
                data = data.reshape(-1, self.embedding_dim)
            return data

    def _load_audio(self, vid, iid, label):
        sent = 'pos' if label == 'humor' else 'neg'
        base = f"{vid}_{sent}_{iid}.npy"
        try:
            arr = self._load_npy('audio', base)
        except Exception:
            arr = self._load_npy('audio', f"_{base}")
        return arr

    def _load_video(self, vid, iid, label):
        sent = 'pos' if label == 'humor' else 'neg'
        filename = f"{vid}_{sent}_{iid}.npy"
        feat_folder = f"open_face_{self.openface_windows}_full"
        return self._load_npy(feat_folder, filename)


    def _get_figurative_label(self, vid, iid):
        if not hasattr(self, '_figurative_map'):
            path = os.path.join(self.features_dir, 'figurative.jsonl')
            pattern = re.compile(r'^"TIENE LENGUAJE FIGURATIVO"')
            figurative_map = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    label = entry['gpt_label']
                    is_figurative = 1 if pattern.match(label) else 0
                    figurative_map[(entry['video_id'], entry['instance_id'])] = is_figurative
            self._figurative_map = figurative_map

        return np.array([ self._figurative_map.get((vid, iid), 0) ])

    def build_dataset(self, feature_types):
        X, y = [], []
        for _, row in self.metadata.iterrows():
            feats = []
            for f in feature_types:
                if f == 'audio':
                    vec = self._load_audio(row['video_id'], row['instance_id'], row['label'])
                elif f == 'video':
                    vec = self._load_video(row['video_id'], row['instance_id'], row['label'])
                else:
                    vec = self.feature_loaders[f](row['video_id'], row['instance_id'])
                vec_flat = np.ravel(vec)
                feats.append(vec_flat)
            sample = np.concatenate(feats)
            X.append(sample)
            y.append(1 if row['label'] == 'humor' else 0)
        return np.vstack(X), np.array(y)

    def run(self, model_name, feature_types, test_size=0.2, random_state=42):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Options: {list(self.models.keys())}")
        X, y = self.build_dataset(feature_types)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        model = self.models[model_name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, target_names=['HUMOR','NO HUMOR']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds, labels=[1,0]))
        return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run humor detection experiments.')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset directory.')
    parser.add_argument('--model', type=str, choices=['svm','xgboost','dense'], default='svm', help='Model to use.')
    parser.add_argument('--features', nargs='+', choices=['text','audio','vad','figurative','video'], default=['text'], help='Feature types to include.')
    parser.add_argument('--embdim', type=int, help='Embedding dimension for raw binary loaders (e.g., 768).')
    parser.add_argument('--openface-windows', type=int, default=5, help='Number of OpenFace windows (e.g., 5, 10, 20, 100).')
    args = parser.parse_args()

    exp = ModelExperiment(
        dataset_dir=args.dataset, 
        embedding_dim=args.embdim,
        openface_windows=args.openface_windows
    )
    exp.run(args.model, args.features)
