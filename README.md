# Credit Card Fraud Detection

## Project Overview

The project is about detecting credit card fraud using machine learning algorithms. The dataset used in this project is from Kaggle and can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

Techonologies used in this project are:
- Python
- Pandas
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- SHAP
- MLflow
- FastAPI
- Docker

## Project Structure

The project is structured as follows:

```
app/
    api_test.sh         # Script to test the API
    app.py              # FastAPI application
    requirements.txt    # Requirements for the FastAPI app
    Dockerfile          # Dockerfile to build the FastAPI app image
data/
    creditcard.csv      # The dataset used in the project
models/
    *.pkl               # Trained models will be saved here
notebooks/
    EDA.ipynb           # Jupyter notebook for exploratory data analysis
    explain.ipynb       # Jupyter notebook for model explanation using SHAP
requirements.txt        # Requirements for the project
train.py                # Script to train the models
README.md
.gitignore
```

## Exploratory Data Analysis

Since the dataset is pre-procesed, we will only perform some basic exploratory data analysis to understand the data better.

After peeking into the data, we can see that the data is highly unbalanced. The target variable is `Class` which is binary. The value `1` represents fraud and `0` represents non-fraud. The dataset contains 31 features, 28 of which are anonymized and the remaining 3 are `Time`, `Amount`, and `Class`.

## Model Training

We will train a few machine learning models to detect credit card fraud. The models we will train are: XGBoost, LightGBM. RandomForest, Logistic Regression and other classification algorithms will not be used in this project since their training time is too long and they do not perform well on this dataset.

Due to the highly unbalanced dataset, we will use stratified k-fold cross-validation to evaluate the models. The dataset will be split into 5 folds, with each fold containing the same proportion of fraud and non-fraud transactions.

Grid search will be used to find the best hyperparameters for each model and the best model will be selected based on the `f1` score.

To deal with the imbalanced dataset, we will use the following techniques:
- Oversampling: SMOTE.
- Undersampling: RandomUnderSampler.

There is a misconception that some people do the sampling before splitting the data into training and testing sets. This is wrong. The correct way is to split the data first and then do the sampling in the cross-validation loop. This is because if we do the sampling before splitting the data, we will be using the testing data to train the model which is wrong.

For tracking the model training process, we will use MLflow.

Please refer to the `train.py` script for more details about training arguments.

## Model Evaluation

The best model will be evaluated on the test set using the following metrics: precision, recall, f1-score, and ROC-AUC; inference time on test set will also be measured (in seconds).

| Model | Precision | Recall | F1 | ROC-AUC | Inference Time (s) |
| --- | --- | --- | --- | --- | --- |
| XGBoost (Without GridSearch, No Sampling) | 0.92 | 0.81 | 0.86 | 0.90 | 0.37 |
| XGBoost (No Sampling) | 0.96 | 0.80 | 0.87 | 0.90 | 0.48 |
| XGBoost (Oversampling) | 0.82 | 0.81 | 0.81 | 0.90 | 0.41 |
| XGBoost (Undersampling) | 0.06 | 0.89 | 0.11 | 0.93 | 0.28 |
| LightGBM (Without GridSearch, No Sampling) | 0.18 | 0.69 | 0.29 | 0.84 | 0.68 |
| LightGBM (No Sampling) | 0.94 | 0.64 | 0.76 | 0.82 | 0.40 |
| LightGBM (Oversampling) | 0.33 | 0.81 | 0.47 | 0.90 | 1.93 |
| LightGBM (Undersampling) | 0.03 | 0.93 | 0.05 | 0.94 | 0.75 |

The best model is XGBoost without sampling. Since the dataset is highly unbalanced, I directly used the sampling methods to deal with the imbalance. However, the best model is the one without sampling. This is because the sampling methods can introduce bias into the model. The best model is the one that can generalize well to unseen data.

Since having a balanced precision and recall is important in this case. Low precision means that the model will flag many non-fraud transactions as fraud. Low recall means that the model will miss many fraud transactions. The F1 score is the harmonic mean of precision and recall. The best model has a high F1 score which means that it has a balanced precision and recall.

## Model Explanation

We will use SHAP to explain the model. For detailed explanation, please refer to the `explain.ipynb` notebook.

## Deployment
The trained model can be deployed using FastAPI. The API will accept a JSON object with the following structure:

```json
{
    "Time": 0,
    "V1": 0.1,
    "V2": 0.2,
    ...
    "V28": 0.3,
    "Amount": 100.0
}
```
The API will return a JSON object with the following structure:

```json
{
    "prediction": 0,
}
```

The FastAPI application can be run using Docker. The Dockerfile is provided in the `app` directory. To build the Docker image, run the following command in the `app` directory:

```bash
docker build -t credit-card-fraud-detection .
```
To run the Docker container, use the following command:

```bash
docker run -p 8000:8000 --rm credit-card-fraud-detection
```

To test the API, you can use the provided `api_test.sh` script. Make sure to run the FastAPI application first.

```bash
./api_test.sh
```

## Conclusion

The best model is XGBoost without sampling. One thing I learned from this project is that a method that is supposed to improve the model can actually make it worse. This is why it is important to do alot of experiments to find the best model.