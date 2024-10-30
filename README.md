# Predicting Problematic Internet Usage Based on Physical Activity

## Project Description
This project develops a model to predict problematic internet use in children and adolescents using physical activity and fitness data. It aims to identify early signs of unhealthy internet habits and provide insights for timely intervention.

## Target Variable
The target variable in this analysis is **ordinal**, meaning it has a natural order but the distances between the categories are not necessarily equal. The target variable ranges from **1 to 4**, where:

- 1: Low problematic internet usage
- 2: Moderate problematic internet usage
- 3: High problematic internet usage
- 4: Severe problematic internet usage

Given the ordinal nature of the target, the model needs to take into account the order of the classes. For this reason, both classification and regression-style metrics are used to evaluate model performance.

## Suggested Metrics

### 1. Quadratic Weighted Kappa (QWK)
Quadratic Weighted Kappa (QWK) measures the agreement between the predicted and actual ordinal labels. It penalizes larger differences more heavily, making it ideal for ordinal classification tasks.

### 2. Accuracy
Accuracy measures the percentage of correct predictions. Although it doesn't account for the ordinal nature of the data, it provides a baseline performance measure.

### 3. Precision, Recall, and F1-Score (Per Class)
These metrics give insight into how well the model predicts each class. 
- **Precision** measures the proportion of true positive predictions for each class.
- **Recall** (or sensitivity) measures how well the model identifies true positives.
- **F1-Score** balances precision and recall to provide a single performance metric per class.

### 4. Confusion Matrix
The confusion matrix provides a summary of prediction results, showing where the model correctly predicts and where it misclassifies, helping to identify patterns in the errors (e.g., whether the model predicts "High" internet usage when the true label is "Moderate").

### 5. Mean Absolute Error (MAE)
MAE is used to assess how far off the model's predictions are from the true ordinal labels. By treating the ordinal classes as continuous values, MAE measures the average magnitude of the errors.

### 6. Cohen’s Kappa (Optional)
Cohen’s Kappa measures agreement between the true and predicted classes while accounting for chance agreement. It is similar to QWK but does not penalize larger misclassifications more heavily.

---

## Conclusion
The model will be evaluated using a combination of these metrics to ensure effective handling of the **ordinal nature** of the target variable. Accuracy and confusion matrices provide an overall picture, while **Quadratic Weighted Kappa** and **Mean Absolute Error** offer deeper insight into how well the model captures the relationships between the ordered classes.
