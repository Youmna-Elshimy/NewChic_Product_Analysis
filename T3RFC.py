# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Google Drive (Replace with your file path if needed)
# Assuming you have the dataset ready
!gdown --id 1CiVjAMtFZoOAVzQS6tOi1I6Fr3AkVzqR  # Uncomment if needed

# Load your preprocessed DataFrame
df = pd.read_csv('/content/Updated_CSV.csv')  # Use your correct file path

# Select relevant columns for features and target
features = ['discount', 'likes_count', 'current_price', 'raw_price']
target = 'category'

# Extract the selected categories
categories = ['accessories', 'bags', 'beauty', 'jewelry', 'kids', 'men', 'women', 'house', 'shoes']
df = df[df[target].isin(categories)]

# Encode target labels to numerical values
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Prepare the data
X = df[features]
y = df[target]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

# Create a copy of the test set for displaying results
df_test = X_test.copy()
df_test['predicted_category'] = le.inverse_transform(y_pred)

# Display overall top 10 products based on likes_count across all categories
def display_overall_top_products(df_test, original_df):
    # Merge test set back with the original data to get full product details
    merged_df = pd.merge(df_test, original_df, on=features)

    # Sort by likes_count and display the overall top 10 products
    top_products = merged_df.sort_values(by='likes_count', ascending=False).head(10)
    print("\nOverall Top 10 Products based on likes_count:")
    print(top_products[['likes_count', 'discount', 'current_price', 'raw_price', 'category', 'predicted_category']])

    return top_products

# Get overall top 10 products and store in a DataFrame
top_10_products_df = display_overall_top_products(df_test, df)

# Plot the top 10 products based on likes_count
plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_products_df['likes_count'], y=top_10_products_df.index, palette='coolwarm')
plt.title("Top 10 Products by Likes Count Across All Categories")
plt.xlabel("Likes Count")
plt.ylabel("Index")
plt.show()

# Function to display top 10 products per category
def display_top_products(df_test, original_df, category):
    # Filter the test set for the specified category
    category_products = df_test[df_test['predicted_category'] == category]

    # Merge the test set back with the original data to get product names
    merged_df = pd.merge(category_products, original_df, on=features)

    # Sort by likes_count and display the top 10
    top_products = merged_df.sort_values(by='likes_count', ascending=False).head(10)
    print(f"\nTop 10 products in {category} category:")
    print(top_products[['likes_count', 'discount', 'current_price', 'raw_price', 'category']])

# Loop through each category and display the top products
for category in categories:
    display_top_products(df_test, df, category)

# Plot the feature importance from the Random Forest model
feature_importances = rf_classifier.feature_importances_
features_names = features

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features_names)
plt.title("Feature Importance in Random Forest Classifier")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Confusion matrix to visualize the performance of the model
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Categories")
plt.ylabel("True Categories")
plt.show()

# Visualizing the distribution of predicted categories
df_test['predicted_category'] = le.inverse_transform(y_pred)
predicted_counts = df_test['predicted_category'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=predicted_counts.index, y=predicted_counts.values)
plt.title("Distribution of Predicted Categories")
plt.xlabel("Categories")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()



# Calculate and Display the Category with the Highest Popularity Based on Selected Metrics

# Group the data by category and calculate the sum/mean for the popularity metrics
popularity_metrics = df.groupby('category')[['likes_count', 'discount', 'current_price', 'raw_price']].mean()

# Calculate a composite popularity score (you can modify the weights here)
popularity_metrics['popularity_score'] = (popularity_metrics['likes_count'] +
                                          (popularity_metrics['discount'] * 0.5) -
                                          popularity_metrics['current_price'] +
                                          popularity_metrics['raw_price'])

# Sort the categories by the computed popularity score
popularity_metrics = popularity_metrics.sort_values(by='popularity_score', ascending=False)

# Use inverse_transform to get the category names back from encoded labels
popularity_metrics.index = le.inverse_transform(popularity_metrics.index)

# Display the category with the highest popularity score
print(f"Category with the highest popularity based on selected metrics: {popularity_metrics.index[0]}")

# Display the entire sorted popularity dataframe
print(popularity_metrics)

# Plot the popularity scores for each category
plt.figure(figsize=(12, 8))
sns.barplot(x=popularity_metrics.index, y=popularity_metrics['popularity_score'], palette='coolwarm')
plt.title("Popularity Score by Category")
plt.xlabel("Category")
plt.ylabel("Popularity Score")
plt.xticks(rotation=45)
plt.show()