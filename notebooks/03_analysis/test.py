# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, classification_report)
import itertools

# Import Data
df = pd.read_pickle(r'data\04_fct\fct_demographic_offers_and_transactions.pkl')

# Define the feature matrix and target variable using the original dataset without filtering
features = ['age', 'income', 'days_as_member', 'gender_F', 'gender_M', 'is_bogo', 'is_discount', 'reward', 'difficulty', 'duration_hrs']
X = df[features]

# Modify the target variable to include only offers that were viewed before being completed
df['offer_completed_viewed'] = df.apply(lambda x: 1 if x['offer_completed'] == 1 and x['viewed_before_completion'] == 1 else 0, axis=1)
y = df['offer_completed_viewed']

# Check the new distribution of the target variable
target_distribution_new = y.value_counts()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Define a new customer profile and offers (for demonstration)
new_customer_profile = pd.DataFrame({
    'age': [30, 40, 50, 60],
    'income': [50000, 60000, 70000, 80000],
    'days_as_member': [200, 400, 600, 800],
    'gender_F': [0, 1, 0, 1],
    'gender_M': [1, 0, 1, 0]
})

offers = pd.DataFrame({
    'is_bogo': [0, 1, 0, 1],
    'is_discount': [1, 0, 1, 0],
    'reward': [2, 5, 3, 10],
    'difficulty': [10, 5, 7, 10],
    'duration_hrs': [168, 120, 240, 168]
})

# Create a combined dataset for prediction
customer_offer_pairs = pd.DataFrame(itertools.product(new_customer_profile.index, offers.index), columns=['customer_idx', 'offer_idx'])
customer_offer_pairs = customer_offer_pairs.merge(new_customer_profile, left_on='customer_idx', right_index=True)
customer_offer_pairs = customer_offer_pairs.merge(offers, left_on='offer_idx', right_index=True)

# Predict response probability
X_new = customer_offer_pairs[features]
customer_offer_pairs['response_probability'] = model.predict_proba(X_new)[:, 1]

# Calculate top recommendations
grouped = customer_offer_pairs.groupby('customer_idx')
sorted_pairs = customer_offer_pairs.sort_values(by=['customer_idx', 'response_probability'], ascending=[True, False])
top_per_group = sorted_pairs.drop_duplicates(subset=['customer_idx'])
top_recommendations = top_per_group.reset_index(drop=True)
top_recommendations

# Predict on the test set
y_test_pred = model.predict(X_test)

### Model Validation Metrics ###
# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'The ROC-ACU score is: {roc_auc}')

# Calculate False Positive and False Negative Rates
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
fpr = fp / (fp + tn)
print(f'The False Positive Rate is: {fpr}')
fnr = fn / (fn + tp)
print(f'The False Negative Rate is: {fnr}')

# Calculate Cross-Validation Score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
mean_cv_score = cv_scores.mean()
print(f'The mean cross-validation F1 score is: {mean_cv_score}')
std_cv_score = cv_scores.std()
print(f'The standard deviation of the cross-validation F1 scores is: {std_cv_score}')

### Generate Classication Report ###
class_report = classification_report(y_test, y_test_pred, output_dict=True)
df_main = pd.DataFrame(class_report).transpose().drop(['accuracy'])
overall_metrics = pd.DataFrame(class_report).transpose().loc[['accuracy']]
df_final = pd.concat([df_main, overall_metrics])
df_final

metrics_dict = {
    'ROC-AUC Score': roc_auc,
    'False Positive Rate': fpr, 
    'False Negative Rate': fnr,
    'F1 Macro Avg': df_final.loc['macro avg', 'f1-score'],
    'F1 Weighted Avg': df_final.loc['weighted avg', 'f1-score'],
    'Mean Cross-Validation F1 Score': mean_cv_score,
    'STD Cross-Validation F1 Score': std_cv_score
}

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
metrics_df.to_csv(r'data/04_fct/fct_personalized_evaluation_results.csv', index=False)
metrics_df