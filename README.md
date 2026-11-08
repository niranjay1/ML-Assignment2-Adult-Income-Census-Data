# ML-Assignment2-Adult-Income-Census-Data
Adult Income Prediction based on census data using Five Machine Learning Classification Algorithms

## a. Problem Statement

The objective of this project is to predict whether an individual's annual income is <=50K or >50K using demographic and employment-related information from the Adult Census Income dataset. 

This dataset, extracted from the 1994 US Census database by Barry Becker and contributed to the UCI Machine Learning Repository in 1996. It contains 14 input features and a binary income target.

Five machine learning classification algorithms are implemented and evaluated to compare their predictive performance and identify the most suitable model. This is a binary classification problem.

## b. Dataset Description

The project uses the Adult Census Income dataset from the UCI Machine Learning Repository.

**Dataset:** Adult Census Income (Census Income)

**Source:** UCI Machine Learning Repository

**Dataset contributors:** Barry Becker and Ronny Kohavi

**Dataset characteristics:**
- Total records: 48,842
- Input features: 14
- Target variable: Income
- Classification type: Binary classification
- Feature types: Numerical and categorical
- Missing values: Present in selected attributes

**Training and testing split used in this project:**
- Training records: 32,561
- Testing records: 16,281

The 14 input features include demographic and employment-related information such as age, workclass, education, marital status, occupation, race, sex, hours per week, capital gain and capital loss.

The target variable contains two categories:
- `<=50K`: Annual income is $50,000 or less.
- `>50K`: Annual income is greater than $50,000.

## c. GitHub Repository Link

https://github.com/niranjay1/ML-Assignment2-Adult-Income-Census-Data

## d. Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Data preprocessing includes:
1. Missing-value imputation
2. Feature scaling for numerical variables
3. One-hot encoding of categorical variables

The preprocessing steps and classifiers are combined using Scikit-learn pipelines. 

### Comparison Table – Evaluation Metrics

All five models were evaluated on the same test dataset using six classification metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 85.07% | 90.38% | 72.82% | 58.71% | 65.01% | 0.5618 |
| Decision Tree | 81.27% | 74.42% | 60.13% | 61.41% | 60.77% | 0.4847 |
| KNN | 83.33% | 85.64% | 66.39% | 59.62% | 62.82% | 0.5224 |
| Gaussian Naive Bayes | 55.11% | 76.11% | 33.78% | 93.76% | 49.67% | 0.3300 |
| SVM | 85.18% | 90.32% | 73.73% | 57.88% | 64.85% | 0.5629 |

### Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved the highest AUC (90.38%) and F1-score (65.01%), indicating good overall classification performance and balance between precision and recall. |
| Decision Tree | Achieved 81.27% accuracy but had the lowest AUC (74.42%) among the five models. |
| KNN | Achieved 83.33% accuracy and 85.64% AUC, demonstrating reasonably good classification performance. |
| Naive Bayes | Achieved the highest recall (93.76%) but low precision (33.78%), indicating a high number of false-positive predictions. |
| Support Vector Machine | Achieved the highest accuracy (85.18%), precision (73.73%) and MCC (0.5629). Its overall performance was very close to Logistic Regression. |
| Overall Winner | Logistic Regression can be considered the best overall choice based on its balanced performance. It achieved the highest AUC (90.38%) and F1-score (65.01%), while its accuracy (85.07%) was very close to the best-performing SVM model (85.18%). SVM also performed very well, achieving the highest accuracy, precision, and MCC. The difference between Logistic Regression and SVM was small. Therefore, Logistic Regression is selected as the overall preferred model based on its combination of classification performance and balance across metrics. |

## e. Application

A Streamlit application has been developed to:
- Upload test data in CSV format.
- Select one of the five trained classification models.
- Evaluate the selected model using six classification metrics.
- Display the confusion matrix and classification report.
- View and download sample predictions.

The application loads previously trained models and does not retrain them during evaluation.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
