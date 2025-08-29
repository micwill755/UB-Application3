# Bank Marketing Campaign Classification

A comprehensive machine learning analysis to predict customer subscription to term deposits based on Portuguese bank marketing campaign data.

## Project Overview

This project analyzes a dataset from direct marketing campaigns of a Portuguese banking institution to predict whether customers will subscribe to a term deposit. The analysis includes data preprocessing, exploratory data analysis, feature engineering, model comparison, and performance optimization.

## Dataset

- **Source**: UCI Machine Learning Repository - Bank Marketing Dataset
- **Size**: 41,176 customer records with 21 features
- **Target**: Binary classification (yes/no subscription to term deposit)
- **Class Distribution**: Highly imbalanced (11.3% positive responses)

### Key Features
- **Customer Demographics**: Age, job, marital status, education
- **Financial Information**: Default status, housing loan, personal loan
- **Campaign Data**: Contact type, duration, number of contacts
- **Economic Indicators**: Employment variation rate, consumer price index, Euribor rate

## Data Preprocessing

### Missing Values & Data Quality
- Handled 'unknown' values across categorical features
- Replaced missing marital status with mode ('married')
- Removed duplicate records
- Validated outliers (age >70, call duration >1000s deemed valid)

### Feature Engineering
- Created `previously_contacted` binary feature from `pdays`
- Applied one-hot encoding for categorical variables
- Standardized numerical features for linear models

## Exploratory Data Analysis

### Key Findings
- **Target Distribution**: 88.7% "no", 11.3% "yes" (highly imbalanced)
- **Contact Success**: Cellular contacts more effective than telephone
- **Seasonal Patterns**: Campaign success varies by month
- **Economic Impact**: Economic indicators strongly correlate with success rates

## Model Development

### Models Evaluated
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree**

### Baseline Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 91.4% | 65.2% | 51.4% | 57.5% |
| SVM | 91.2% | 66.9% | 43.2% | 52.5% |
| Logistic Regression | 91.1% | 66.0% | 42.7% | 51.8% |
| KNN | 90.9% | 64.6% | 41.8% | 50.8% |

## Dimensionality Reduction (PCA)

### Principal Component Analysis Results
- **Components for 95% variance**: 24 out of 53 features
- **Key themes identified**:
  - **PC1 (27.8% variance)**: Economic indicators (euribor3m, employment rates)
  - **PC2 (12.6% variance)**: Customer contact history (previous contacts, days since last contact)
  - **PC3 (7.9% variance)**: Demographics (age, consumer confidence)

### PCA Impact on Performance
- **Computational Efficiency**: Faster training for linear models
- **Performance**: Maintained or slightly improved test accuracy
- **Feature Reduction**: 53 → 24 dimensions while retaining 95% information

## Model Optimization

### Hyperparameter Tuning
Applied GridSearchCV with 5-fold cross-validation:

- **Logistic Regression**: C values, solvers
- **SVM**: C values, kernels (linear, RBF)
- **KNN**: Number of neighbors, distance weighting
- **Decision Tree**: Max depth, min samples split

### Final Performance (After Tuning)
- **Best F1-Score**: Decision Tree (0.575)
- **Best ROC-AUC**: Logistic Regression (0.935)
- **Best Recall**: Decision Tree (0.514)

## Key Insights

### Business Intelligence
1. **Economic Conditions**: Strongest predictor of subscription success
2. **Contact Strategy**: Previous contact history significantly impacts outcomes
3. **Customer Demographics**: Age and consumer confidence are important factors
4. **Campaign Timing**: Economic climate affects campaign effectiveness

### Technical Insights
1. **Class Imbalance**: Accuracy misleading; focus on recall and F1-score
2. **Feature Importance**: Economic indicators > Contact history > Demographics
3. **Model Selection**: Decision Tree best for recall; Logistic Regression best for discrimination
4. **Dimensionality**: 24 components sufficient for 95% information retention

## Model Recommendations

### For Maximum Recall (Catch More Subscribers)
- **Model**: Decision Tree
- **Threshold**: Adjust below 0.5 to increase sensitivity
- **Use Case**: When missing potential customers is costly

### For Balanced Performance
- **Model**: Logistic Regression
- **Advantage**: Best ROC-AUC (0.935) for threshold flexibility
- **Use Case**: When precision-recall balance is important

## Files Structure

```
├── README.md
├── prompt_III.ipynb          # Main analysis notebook
├── data/
│   └── bank-additional-full.csv
└── requirements.txt
```

## Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook prompt_III.ipynb`

## Results Summary

This analysis successfully demonstrates that economic conditions are the primary driver of term deposit subscriptions, followed by customer contact history and demographics. The Decision Tree model achieved the best recall (51.4%) for identifying potential subscribers, while Logistic Regression provided the best overall discriminative ability (ROC-AUC: 0.935). The PCA analysis revealed that 24 carefully selected feature combinations can capture 95% of the data's predictive power, enabling more efficient model training without sacrificing performance.

## Future Work

- Implement ensemble methods (Random Forest, Gradient Boosting)
- Explore advanced feature engineering techniques
- Investigate cost-sensitive learning approaches
- Develop real-time prediction pipeline
- A/B testing framework for campaign optimization

---

*This project was completed as part of UC Berkeley's Machine Learning course, demonstrating end-to-end machine learning workflow from data preprocessing through model deployment recommendations.*
