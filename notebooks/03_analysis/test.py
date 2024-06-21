import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import data
df = pd.read_pickle(r'data\04_fct\fct_demographic_offers_and_transactions.pkl')

### Segment Customers ###
# Extract demographic features for clustering
demographic_features = df[['age', 'income', 'days_as_member', 'gender_F', 'gender_M']]

# Standardize the features
scaler = StandardScaler()
demographic_features_scaled = scaler.fit_transform(demographic_features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(demographic_features_scaled)

# Display the first few rows with the segment labels
df[['age', 'income', 'days_as_member', 'gender_F', 'gender_M', 'segment']].head()

### Segment Customers ###
# Extract demographic features for clustering
demographic_features = df[['age', 'income', 'days_as_member', 'gender_F', 'gender_M']]

# Standardize the features
scaler = StandardScaler()
demographic_features_scaled = scaler.fit_transform(demographic_features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(demographic_features_scaled)

# Display the first few rows with the segment labels
df[['age', 'income', 'days_as_member', 'gender_F', 'gender_M', 'segment']].head()

# Calculate mean values of features for each segment
cluster_characteristics = df.groupby('segment')[['age', 'income', 'days_as_member', 'gender_F', 'gender_M']].mean()
cluster_characteristics['num_cust'] = df.groupby('segment').size()
cluster_characteristics['perc_cust'] = (cluster_characteristics['num_cust'] / df.shape[0]) * 100

# Display the characteristics of each cluster
cluster_characteristics = round(cluster_characteristics,2)
cluster_characteristics

# Reset the index to include 'customer_id' as a column
df_reset = df.reset_index()

response_data = df_reset.groupby(['segment', 'is_bogo', 'is_discount', 'reward', 'difficulty', 'duration_hrs']).agg(
    {
    'customer_id': 'nunique',
    'offer_viewed': 'mean',
    'viewed_before_completion': 'mean',
    'offer_completed': ['mean', 'sum'],
    'total_transactions': ['sum', 'median'],
    'total_transaction_amount': ['sum', 'median']  # Added 'median' aggregation
    }).reset_index()

# Flatten the MultiIndex columns
response_data.columns = ['_'.join(col).strip('_') for col in response_data.columns.values]

# Rename columns for clarity, including the new customer count and median total transaction amount columns
response_data.rename(columns={
    'customer_id_nunique': 'num_customers',
    'offer_viewed_mean': 'viewed_rate', 
    'viewed_before_completion_mean': 'viewed_before_completion_rate',
    'offer_completed_mean': 'completion_rate',
    'offer_completed_sum': 'offers_completed',
    'total_transactions_sum': 'total_transactions',
    'total_transactions_median': 'median_total_transactions',
    'total_transaction_amount_sum': 'total_transaction_amount',
    'total_transaction_amount_median': 'median_total_transaction_amount',
    }, inplace=True)

rates = ['viewed_rate','viewed_before_completion_rate', 'completion_rate','total_transaction_amount']
response_data[rates] = round(response_data[rates] * 100, 2)

response_data.to_csv(r'data\04_fct\fct_segmented_offer_responses.csv')
response_data.to_pickle(r'data\04_fct\fct_segmented_offer_responses.pkl')
response_data.head()

def calculate_score(row, medians):
    score = 0
    # Criteria scoring
    score += row['num_customers'] > medians['num_customers']
    score += row['viewed_rate'] > medians['viewed_rate']
    score += row['viewed_before_completion_rate'] > medians['viewed_before_completion_rate']
    score += row['completion_rate'] > medians['completion_rate']
    score += row['median_total_transactions'] < medians['median_total_transactions']
    score += row['median_total_transaction_amount'] > medians['median_total_transaction_amount']
    return score

def get_optimal_rows(df, segment, top_n=None):
    seg_df = df[df['segment'] == segment].copy()
    medians = seg_df.median()
    
    # Apply score calculation for each row
    seg_df.loc[:, 'score'] = seg_df.apply(lambda row: calculate_score(row, medians), axis=1)
    
    # Sort by score in descending order to get rows with the highest scores at the top
    if top_n is None:
        optimal_rows = seg_df.sort_values(by='score', ascending=False)
    else:
        optimal_rows = seg_df.sort_values(by='score', ascending=False).head(top_n)
    
    return optimal_rows

# Concatenate top rows for each segment
response_scores = pd.concat([get_optimal_rows(response_data, i, top_n=None) for i in range(3)])
response_scores.to_csv(r'data\04_fct\fct_segmented_offer_response_scores.csv')
response_scores.to_pickle(r'data\04_fct\fct_segmented_offer_response_scores.pkl')
response_scores.head()

top_2 = pd.concat([get_optimal_rows(response_data, i, top_n=2) for i in range(3)])
top_2