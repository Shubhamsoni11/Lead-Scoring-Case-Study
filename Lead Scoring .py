#!/usr/bin/env python
# coding: utf-8

# In[146]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings  # Supress Warnings
warnings.filterwarnings('ignore')
sns.set(context="notebook", palette="Spectral", style='darkgrid', font_scale=1.5, color_codes=True)


# In[91]:


#Loading data set
df = pd.read_csv("Leads.csv")
df


# In[92]:


df.info()


# In[93]:


df.describe()


# In[94]:


#checking missing %tage
100*df.isnull().mean()


# In[95]:


df.columns


# In[96]:


df['Lead Quality'].unique()


# In[97]:


df = df.loc[:, df.isnull().mean() < 0.35] #Droping Columns with more the 35% Null Values


# In[98]:


100*df.isnull().mean() #checking missing %tage


# In[99]:


# Drop rows where 'City' is NaN
df_dropped = df.dropna(subset=['City'])

# Verify the changes
print(df_dropped['City'].value_counts(dropna=False))


# In[100]:


# Plot histogram for numerical features
numerical_cols = df_dropped.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(14, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_dropped[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[101]:


# Plot box plots for numerical features
plt.figure(figsize=(14, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_dropped[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


# In[102]:


def countplot(x, fig):
    plt.subplot(2, 2, fig)
    sns.countplot(data=df, y=x, order= df[x].value_counts().index)
    plt.title('Count across ' + x, size=16)
    plt.xlabel('Count', size=14)
    plt.ylabel(x, size=14)
    plt.xticks(rotation=90)

plt.figure(figsize=(15, 10))
countplot('How did you hear about X Education', 1)
countplot('Lead Profile', 2)
countplot('Specialization', 3)

plt.tight_layout()
plt.show()


# In[103]:


df['Lead Source'].fillna(df['Lead Source'].mode()[0], inplace=True)

# Fill missing values in 'TotalVisits' and 'Page Views Per Visit' with the median
df['TotalVisits'].fillna(df['TotalVisits'].median(), inplace=True)
df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median(), inplace=True)

# Plot city-wise count of leads
plt.figure(figsize=(12, 6))
city_counts = df['City'].value_counts()
city_counts.plot(kind='bar', color='skyblue')
plt.title('City-wise Lead Count')
plt.xlabel('City')
plt.ylabel('Count of Leads')
plt.xticks(rotation=45)
plt.show()


# In[104]:


# Plot lead source-wise count of leads as a pie chart with percentage and table
plt.figure(figsize=(14, 7))

# Plot pie chart
plt.subplot(1, 2, 1)
lead_source_counts.plot(kind='bar', color ='green')
plt.title('Lead Source-wise Lead Count')
plt.ylabel('')  # Hide the y-label for better presentation

# Plot table
plt.subplot(1, 2, 2)
plt.axis('off')
tbl = plt.table(cellText=lead_source_counts.reset_index().values,
                colLabels=['Lead Source', 'Count'],
                cellLoc='center',
                loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)

plt.show()


# In[105]:


plt.figure(figsize=(10,5))
s1 = sns.countplot(x='City', hue='Converted', data= df)  # Ensure 'your_dataframe' is the DataFrame you're using
s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
plt.show()


# In[106]:


total_leads_received = len(df)
total_converted = df['Converted'].sum()  # Assuming 'Converted' is 1 for converted and 0 for not

percentage_converted = (total_converted / total_leads_received) * 100

print(f"Total Leads Received: {total_leads_received}")
print(f"Total Leads Converted: {total_converted}")
print(f"Percentage of Leads Converted: {percentage_converted:.2f}%")


# In[107]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= df).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')
plt.show()


# In[108]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= df).tick_params(axis='x', rotation = 90)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= df).tick_params(axis='x', rotation = 90)
plt.title('Do Not Call')
plt.show()


# In[109]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= df).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')

plt.show()


# In[110]:


numeric = df[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# In[111]:


df.drop(['Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'], axis = 1, inplace = True)


# In[112]:


# Find pairs of features with high correlation
threshold = 0.8
high_corr_pairs = [(x, y) for x in corr_matrix.columns for y in corr_matrix.columns 
                   if x != y and abs(corr_matrix.loc[x, y]) > threshold]

print("Highly correlated pairs (absolute correlation > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {corr_matrix.loc[pair[0], pair[1]]}")


# In[113]:


# Drop rows where 'City' is NaN
df = df.dropna(subset=['City'])

# Replace 'Select' entries in the 'City' column with NaN
df['City'] = df['City'].replace('Select', pd.NA)

# Drop rows where 'City' is NaN again after replacing 'Select'
df = df.dropna(subset=['City'])

# Encoding categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Calculate the correlation matrix
corr_matrix = df_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('Correlation Heatmap', size=16)
plt.show()

# Find pairs of features with high correlation
threshold = 0.8
high_corr_pairs = [(x, y) for x in corr_matrix.columns for y in corr_matrix.columns 
                   if x != y and abs(corr_matrix.loc[x, y]) > threshold]

print("Highly correlated pairs (absolute correlation > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {corr_matrix.loc[pair[0], pair[1]]}")


# # Dummy variable creation
# 
# The next step is to dealing with the categorical variables present in the dataset. So first take a look at which variables are actually categorical variables.

# In[127]:


# Checking the columns which are of type 'object'

temp = df.loc[:, df.dtypes == 'object']
temp.columns


# In[135]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(df[['Lead Origin', 'Specialization', 'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation', 'Last Notable Activity']], drop_first=True)

# Add th
e results to the master dataframe
df_final_dum = pd.concat([df, dummy], axis=1)

# Drop the original columns that have been one-hot encoded
df_final_dum = df_final_dum.drop(columns=['Lead Origin', 'Specialization', 'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation', 'Last Notable Activity'])

df_final_dum


# # Linear Regression Model

# In[142]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting the data into training and testing sets
X = df_encoded.drop('Converted', axis=1)
y = df_encoded['Converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[165]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# Building the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Making predictions
y_pred = log_reg.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
print(lr_model.summary())


# In[149]:


from sklearn.feature_selection import RFE

# Scaling the numerical features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize RFE with a logistic regression estimator
rfe = RFE(estimator=log_reg, n_features_to_select=10, step=1)

# Fit RFE on the training data
rfe.fit(X_train_scaled, y_train)

# Get selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:")
print(selected_features)

# Transform training and test sets to include only selected features
X_train_selected = rfe.transform(X_train_scaled)
X_test_selected = rfe.transform(X_test_scaled)

# Retrain the model on selected features
log_reg.fit(X_train_selected, y_train)

# Evaluate the model on selected features
y_pred_selected = log_reg.predict(X_test_selected)

print("\nConfusion Matrix (Selected Features):")
print(confusion_matrix(y_test, y_pred_selected))

print("\nClassification Report (Selected Features):")
print(classification_report(y_test, y_pred_selected))


# In[152]:


# Initialize the logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training data
log_reg.fit(X_train_scaled, y_train)


# In[153]:


# Assuming 'log_reg' is your trained Logistic Regression model
coefficients = pd.DataFrame(log_reg.coef_[0], index=X.columns, columns=['Coefficient'])
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()

# Top three variables contributing most towards conversion probability
top_three_variables = coefficients.sort_values(by='Abs_Coefficient', ascending=False).head(3)
print("Top Three Variables Contributing to Conversion Probability:")
print(top_three_variables)


# In[ ]:




