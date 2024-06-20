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

### Calculate Completion Rates by Offer Type and Segement ###
df_filtered = df[df['viewed_before_completion'] == 1]

# Group data by segment and offer attributes, then calculate the response rate
response_data = df.groupby(['segment', 'is_bogo', 'is_discount', 'reward', 'difficulty', 'duration_hrs']).agg({
    'offer_viewed': 'mean',
    'offer_completed': 'mean'
}).reset_index()

# Rename columns for clarity
response_data.rename(columns={'offer_viewed': 'viewed_rate', 'offer_completed': 'completion_rate'}, inplace=True)

# Identify the top segments for each offer type
top_segments = response_data.sort_values(by='completion_rate', ascending=False).groupby(['is_bogo', 'is_discount', 'reward', 'difficulty', 'duration_hrs']).head(1)

top_segments