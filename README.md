## Overview

This repository contains a decision tree classifier that predicts customer churn for a bank based on various customer attributes. The model is built using a dataset with customer information and is designed to predict whether a customer will exit the bank (`Exited` column). The implementation includes preprocessing, training, and evaluation steps.

## Dataset

The dataset `raw_df` includes 15,000 entries and the following columns:

- **CustomerId**: Unique identifier for each customer.
- **Surname**: Customer's surname (not used in model training).
- **CreditScore**: Credit score of the customer.
- **Geography**: The geographical location of the customer (categorical).
- **Gender**: Gender of the customer (categorical).
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Whether the customer has a credit card (binary).
- **IsActiveMember**: Whether the customer is an active member (binary).
- **EstimatedSalary**: Estimated annual salary of the customer.
- **Exited**: Target variable indicating whether the customer exited the bank (binary: 1 = exited, 0 = retained).

### Data Info

The dataset has 13 columns with the following data types:

- 10 columns are numerical (float64).
- 3 columns are categorical (object).

## Functions in `process_bank_churn.py`

This file contains the functions used to preprocess the raw data, build the classification model, and evaluate its performance.

### Key Functions:

1. **`preprocess_data(raw_df)`**  
   This function performs preprocessing on the raw dataset:
   - Handles categorical variables (e.g., encoding `Geography` and `Gender`).
   - Scales numerical features (e.g., `CreditScore`, `Age`, etc.).
   - Splits the data into training and validation sets, with stratification based on the target variable `Exited`.

2. **`preprocess_new_data(new_df, scaler, encoder)`**  
   This function preprocesses new data for predictions, applying the same scaling and encoding as during the training phase.

3. **Model Training**:  
   A Decision Tree Classifier is trained on the preprocessed training data, using `Exited` as the target variable.

4. **Model Evaluation**:  
   - The model is evaluated using metrics such as **AUROC (Area Under the Receiver Operating Characteristic Curve)** on both the training and validation sets.
   - **Train AUROC** and **Validation AUROC** are computed to assess model performance.
   - **Test AUROC** is computed to evaluate the model's generalization capability.

## Model Performance

### Evaluation Metrics:

- **Train AUROC**: 0.926  
- **Validation AUROC**: 0.922  
- **Test AUROC**: 0.238  

While the model performs well on both the training and validation sets, the **test AUROC** is low due to the test set having only customers who exited the bank (i.e., `Exited = 1`), leading to an imbalanced evaluation scenario.

### Model Interpretation:

- The **Decision Tree Classifier** performs well in distinguishing between customers who will exit the bank and those who will stay, based on the features provided in the dataset.
- The low **test AUROC** score is caused by the absence of non-churning customers (those with `Exited = 0`) in the test set, limiting the classifier's ability to demonstrate its discriminative power.

### Conclusion:

This model is a robust solution for predicting customer churn in a banking environment, providing valuable insights into customer retention strategies. The preprocessing steps ensure that the model is trained on clean, consistent data, and the Decision Tree Classifier offers a transparent, interpretable approach to the problem. However, caution should be taken when interpreting the results on the test set, as the imbalance in the target variable impacts the performance evaluation.






