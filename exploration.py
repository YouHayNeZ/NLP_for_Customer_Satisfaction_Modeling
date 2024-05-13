import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data/ryanair_reviews.csv")
print(data.head(), data.dtypes)

# Change columns
data_cleaned = data.drop(columns=['Unnamed: 0'])
data_cleaned['Date Published'] = pd.to_datetime(data_cleaned['Date Published'])
data_cleaned['Date Flown'] = pd.to_datetime(data_cleaned['Date Flown'], errors='coerce')  # Using coerce to handle any invalid dates
data['Recommended'] = data['Recommended'].map({'yes': True, 'no': False})
data_cleaned['Trip_verified'] = data_cleaned['Trip_verified'].replace({'NotVerified': 'Not Verified', 'Unverified': 'Not Verified'})

# Check for missing values
missing_values = data_cleaned.isnull().sum()
missing_percentage = missing_values[missing_values > 0] / data_cleaned.shape[0] * 100
print(missing_percentage)


# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Categorical Data Analysis: Count plots for various categorical columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
sns.countplot(data=data, x='Trip_verified', ax=axes[0, 0])
sns.countplot(data=data, y='Passenger Country', order=data['Passenger Country'].value_counts().index[:10], ax=axes[0, 1])
sns.countplot(data=data, x='Type Of Traveller', ax=axes[1, 0])
sns.countplot(data=data, x='Recommended', ax=axes[1, 1])
sns.histplot(data=data, x='Overall Rating', bins=10, kde=True, ax=axes[2, 0])
sns.histplot(data=data, x='Seat Comfort', bins=6, kde=True, ax=axes[2, 1])
plt.tight_layout()
plt.show()

# Plotting the histogram and kde of the numerical/categorical columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
sns.histplot(data=data, x='Cabin Staff Service', bins=6, kde=True, ax=axes[0, 0])
sns.histplot(data=data, x='Food & Beverages', bins=6, kde=True, ax=axes[0, 1])
sns.histplot(data=data, x='Ground Service', bins=5, kde=True, ax=axes[0, 2])
sns.histplot(data=data, x='Value For Money', bins=6, kde=True, ax=axes[1, 0])
sns.histplot(data=data, x='Inflight Entertainment', bins=5, kde=True, ax=axes[1, 1])
sns.histplot(data=data, x='Wifi & Connectivity', bins=5, kde=True, ax=axes[1, 2])
plt.tight_layout()
plt.show()


# Correlation matrix to see numerical relationships (pearson)
correlation_matrix = data[['Overall Rating', 'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j + 0.5, i + 0.5, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.title(' Pearson Correlation Matrix')
plt.show()

# Correlation matrix to see numerical relationships (spearman)
correlation_matrix = data[['Overall Rating', 'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service']].corr(method='spearman')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j + 0.5, i + 0.5, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.title(' Spearman Correlation Matrix')
plt.show()

# Compute Cramer's V for categorical variables
