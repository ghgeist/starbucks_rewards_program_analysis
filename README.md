<p align="center">
  <img src="assets\Starbucks-Logo.png"
</p>

# Starbucks Capstone Challenge

# Project Overview
The Starbucks Capstone Challenge aims to analyze customer behavior on the Starbucks rewards mobile app. The goal is to understand how different demographic groups respond to various types of promotional offers and optimize offer distribution strategies. This project uses simulated data that mimics real customer interactions with the Starbucks rewards app, reflecting how individuals make purchasing decisions and how these decisions are influenced by promotional offers.

# Installation and Setup

## Codes and Resources Used
- **Editor:** VSCode
- **Python Version:** 3.12

## Python Packages Used
## Python Packages Used
- **General Purpose:** 
  - `numpy`
  - `pandas`

- **Data Manipulation:**
  - `scikit-learn`

- **Data Visualization:**
  - `matplotlib`
  - `seaborn`
  - `plotly`

# Data
### Portfolio (Ad Campaign Portfolio)
- **Processing**: Offer types and channels were featurized using one-hot encoding. The ‘duration’ column was converted from days to hours.
- **Analysis**: The portfolio dataset (10 offers x 6 fields) contains details for the 10 unique offers sent during a 30-day period. The offers are categorized into three types: BOGO (buy one get one free), informational, and discount. Offers were delivered via email, mobile, social, and web channels with spending prompts ranging from $0 to $20 and rewards from $0 to $10.

### Profile (Customer Profiles)
- **Processing**: The 'became_member_on' column was converted to ‘days_as_member’. Age outliers (i.e., 118) were replaced with NaN, and gender = ‘None’ was replaced with ‘Unknown’. The gender data was then one-hot encoded.
- **Analysis**: The profile dataset (17,000 users x 5 fields) includes customer ID, rewards membership start date, and basic demographic information such as age, income, and gender. Of the 17,000 users, 87.21% (14,813) provided demographic data.

### Transcription (Customer Offers and Transactions)
- **Processing**: The transcript data is an event log dataset from a JSON file. The ‘value’ column was expanded from nested dictionary entries to separate columns. The resulting dataset was then split into two: offers and transactions.
  - **Offers**: Offer events were pivoted out with 'time_hrs' as the values. The aggregation function 'first' was used to get the first occurrence time of each event type for each offer-customer pair. A boolean was also created to determine if the offer was viewed before complete. The portfolio data was joined to offers to make the offers dataset easier to understand.
  - **Transactions**: Unneeded offer columns were dropped from the transactions dataframe.

## Data Preprocessing
The data was processed by merging transactions with offers by 'customer_id' and filtering on those within the offer period. The total number and amount of transactions per customer and offer were then aggregated. The resulting data was merged with the original offers and customer profiles to form a comprehensive dataset. Missing values were filled in, and data columns were converted to the correct data type and downcasted when appropriate.

## Understanding Offer Responses

### Rationale
Customer segments were created to understand how different demographics respond to offers. By analyzing the preferences and behaviors of these segments, marketing strategies can be tailored more effectively.

### Methodology
Demographic features were standardized and grouped into three clusters using the K-Means algorithm, guided by the silhouette score to determine the optimal number of clusters. Offer responses were then aggregated for each segment and offer. "Overspend" (median transaction amount - (reward + difficulty)) was calculated as a potential profitability indicator.

### Results
- **Customer Segmentation**: Three segments were identified: Middle-aged Males, Economistas (younger, lower-income female customers), and Affluent Matriarchs (older, wealthier female customers).
- **Top Segment Offer Responses**: Discounts with a low reward, medium difficulty, and long duration had the best balance of response rates across all segments. The Affluent Matriarchs cluster showed the largest overspend, suggesting a higher profitability potential.

## Predicting Personalizing Recommendation Responses

### Rationale
Understanding customer response patterns allows for better personalization of recommendations. By anticipating customer behaviors and preferences, better customer experiences, higher retention rates, and increased profitability can be achieved.

### Methodology
A Random Forest classifier was used to predict customer responses based on demographic features (age, income, days as a member, gender) and offer characteristics (offer type, reward, difficulty, duration). The model's parameters were optimized via GridSearch.

### Results
The initial model performed reasonably well with a ROC-AUC score of 0.720. The optimized parameters improved all metrics except for the false positive rate. The mean cross-validation score indicated consistent performance across different data subsets.

| Metric                      | Default Value | Optimized Value | Percent Difference |
|-----------------------------|---------------|-----------------|--------------------|
| ROC-AUC Score               | 0.720         | 0.759           | 5.34               |
| False Positive Rate         | 0.270         | 0.298           | 10.49              |
| False Negative Rate         | 0.404         | 0.320           | -20.81             |
| Mean Cross-Validation F1 Score | 0.613         | 0.664           | 8.35               |
| STD Cross-Validation F1 Score  | 0.006         | 0.005           | -15.99             |
| F1 Macro Avg                | 0.663         | 0.690           | 3.96               |
| F1 Weighted Avg             | 0.668         | 0.692           | 3.62               |


## Conclusion

The optimized model demonstrates a robust ability to predict customer responses to promotional offers, providing actionable insights for targeted marketing strategies. The detailed evaluation of the model’s parameters and the validation process ensures that the final model is reliable and effective.

## Reflection and Improvement

One interesting aspect of the project was the significant impact of parameter tuning on the model's performance, highlighting the importance of optimization in machine learning. A challenging aspect was managing the trade-off between false positive and false negative rates. Future improvements could explore additional features or alternative algorithms, such as ensemble learning or deep learning models, to handle more complex patterns in the data.

## Acknowledgements
- Udacity for providing the project framework.
- Starbucks for the simulated data.


# License
[MIT License](https://opensource.org/license/mit/)
