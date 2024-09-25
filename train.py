import argparse
import timeit
import random
import os

import pandas as pd
import numpy as np
import pickle

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, \
                            confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.drop('Class', axis=1)
    y = data['Class']

    return X, y

def evaluate(y_true, y_pred, test=False):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    post_fix = '_test' if test else ''

    return {
        f'Precision{post_fix}': precision,
        f'Recall{post_fix}': recall,
        f'F1{post_fix}': f1,
        f'ROC_AUC{post_fix}': roc_auc
    }

def main(args):
    seed_everything(args.seed)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    model_params = {
        'XGBoostClassifier': {
            'model': XGBClassifier(seed=args.seed),
            'params': {
                'estimator__booster': ['gbtree', 'gblinear', 'dart'],
                'estimator__device': ['gpu'],
                'estimator__eta': [0.01, 0.1, 0.3],
                'estimator__max_depth': [10, 20, 50],
                'estimator__min_child_weight': [1, 5, 10],
                'estimator__subsample': [0.5, 0.7, 1.0],
                'estimator__colsample_bytree': [0.5, 0.7, 1.0],
                'estimator__objective': ['binary:logistic']
            }
        },
        'LightGBMClassifier': {
            'model': LGBMClassifier(seed=args.seed),
            'params': {
                'estimator__boosting_type': ['gbdt', 'dart'],
                'estimator__num_leaves': [20, 30, 40],
                'estimator__max_depth': [-1, 10, 50, 100],
                'estimator__min_gain_to_split': [0.0, 0.1, 0.3],
                'estimator__learning_rate': [0.01, 0.1, 0.3],
                'estimator__n_estimators': [50, 100, 200],
                'estimator__objective': ['binary']
            }
        }
    }

    col_pl = ColumnTransformer([
        ('scaler', StandardScaler(), X.columns)
    ])

    for model_name, model in model_params.items():
        print(f'Training {model_name}')

        pipeline = Pipeline([
            ('preprocessor', col_pl),
            ('sampler', SMOTE()),
            ('estimator', model['model'])
        ])

        with mlflow.start_run(run_name=args.run_name):
            grid_search = GridSearchCV(estimator=pipeline, param_grid=model['params'], scoring='f1', \
                                       cv=StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True), n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)

            prediction_time_start = timeit.default_timer()
            y_train_pred = grid_search.predict(X_train)
            y_test_pred = grid_search.predict(X_test)
            prediction_time_end = timeit.default_timer()

            train_metrics = evaluate(y_train, y_train_pred)
            test_metrics = evaluate(y_test, y_test_pred, test=True)

            confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, normalize='all')
            roc_curve_display = RocCurveDisplay.from_predictions(y_test, y_test_pred)

            signature = infer_signature(model_input=X_test, model_output=y_test_pred, params=grid_search.best_params_)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)
            mlflow.log_metric('prediction_time', prediction_time_end - prediction_time_start)
            mlflow.log_figure(confusion_matrix_display.figure_, 'confusion_matrix.png')
            mlflow.log_figure(roc_curve_display.figure_, 'roc_curve.png')
            mlflow.sklearn.log_model(grid_search.best_estimator_, model_name, signature=signature)

            if args.model_path:
                pickle.dump(grid_search.best_estimator_, open(f'{args.model_path}/{model_name}.pkl', 'wb'))

        print(f'{model_name} trained successfully')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detection Training Script')
    parser.add_argument('--data-path', type=str, help='Path to the training data', default='data/creditcard.csv')
    parser.add_argument('--model-path', type=str, help='Path to the model', default=None)
    parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI', default='http://localhost:5000')
    parser.add_argument('--experiment-name', type=str, help='MLflow experiment name', default='fraud_detection')
    parser.add_argument('--run-name', type=str, help='MLflow run name', default=None)
    parser.add_argument('--seed', type=int, help='Seed for reproducibility', default=42)
    args = parser.parse_args()

    main(args)
