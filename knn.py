import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

# 1. Data Preprocessing
# Load the preprocessed data
data = pd.read_csv('Updated_CSV.csv')

# Select relevant features
features = ['discount', 'likes_count', 'current_price', 'raw_price']
X = data[features]

# Encode the target variable 'category'
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])
y = data['category_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. ML Model
# Build the KNN classifier
clf_knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
clf_knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy of the KNN model: {accuracy:.4f}')

# 3. Top 10 from each category
def display_top_products(data, top_n=10):
    categories = data['category'].unique()
    for category in categories:
        top_products = data[data['category'] == category].nlargest(top_n, 'likes_count')
        print(f"\nTop {top_n} products in category '{category}':")
        print(top_products[['discount', 'likes_count', 'current_price', 'raw_price', 'subcategory']])

# Display top 10 products per category
display_top_products(data)

# 4. Top 10 Products
def display_top_10_products(data, top_n=10):
    top_products = data.nlargest(top_n, 'likes_count')
    print(f"\nTop {top_n} products irrespective of category:")
    print(top_products[['category', 'discount', 'likes_count', 'current_price', 'raw_price', 'subcategory']])

# Display the top 10 products 
display_top_10_products(data)

# 5. Confusion Matrix
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with adjusted figure size and label settings
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Rotate x-axis labels and adjust font size for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Adjust plot title and layout
plt.title('Confusion Matrix', fontsize=16)
plt.tight_layout()  # Ensure the labels don't get cut off
plt.show()

# 6. Best Category Based on Model Accuracy
# The best category can be inferred by checking where the model predicts most accurately in the confusion matrix
category_accuracy = cm.diagonal() / cm.sum(axis=1)
best_category_index = np.argmax(category_accuracy)
best_category = label_encoder.inverse_transform([best_category_index])[0]
print(f'\nThe best category according to the model is: {best_category}')

# 7: Visualize accuracy for different k values
k_values = range(1, 30)
accuracies = []

for k in k_values:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_train, y_train)
    y_pred = clf_knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

accuracies = np.array(accuracies)

# Plot accuracy vs k value
plt.figure(figsize=(10, 6))
plt.errorbar(k_values, accuracies, yerr=np.std(accuracies), fmt='-o', capsize=5)
plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Define a range of k values to test
k_values = range(1, 30)
accuracies = []

# Iterate over different values of k
for k in k_values:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    clf_knn.fit(X_train, y_train)
    y_pred = clf_knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 8. Grid Search
# Define the parameter grid for k values
parameter_grid = {'n_neighbors': range(1, 30, 5)}

# Initialize the KNN classifier
knn_clf = KNeighborsClassifier()

# Initialize GridSearchCV with 10-fold cross-validation
gs_knn = GridSearchCV(knn_clf, parameter_grid, scoring='accuracy', cv=KFold(n_splits=10, shuffle=True))

# Fit the model
gs_knn.fit(X_train, y_train)

# Best k value
best_k = gs_knn.best_params_['n_neighbors']
best_score = gs_knn.best_score_

print(f'\nBest K value: {best_k}')
print(f'Best accuracy from GridSearchCV: {best_score:.4f}')

# Plot the relationship between k and accuracy from GridSearchCV
cv_scores_means = gs_knn.cv_results_['mean_test_score']
cv_scores_stds = gs_knn.cv_results_['std_test_score']
k_range = parameter_grid['n_neighbors']

plt.figure(figsize=(15, 10))
plt.errorbar(k_range, cv_scores_means, yerr=cv_scores_stds, marker='o', label='GridSearchCV Accuracy', capsize=5)
plt.errorbar(k_values, accuracies, yerr=np.std(accuracies), marker='x', label='Manual Accuracy', capsize=5)
plt.ylim([0.1, 1.1])
plt.xlabel('$K$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Comparison of Manual and GridSearchCV KNN Accuracy')
plt.grid(True)
plt.show()