# Import necessary libraries for data handling, models, and visualization
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Load training and test datasets for the wildfire prediction task
train_data = pd.read_csv('wildfires_training.csv')
test_data = pd.read_csv('wildfires_test.csv')

# Preview the first few rows of both datasets to ensure they loaded correctly
print(train_data.head())
print(test_data.head())

# Split data into features (X) and target variable (y) for both training and test sets
X_train = train_data.drop('fire', axis=1)  # Features only
y_train = train_data['fire']  # Target column
X_test = test_data.drop('fire', axis=1)  # Test features
y_test = test_data['fire']  # Test target

# DECISION TREE MODEL

# Initialize Decision Tree classifier with default hyperparameters
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree using the training data
dt_model.fit(X_train, y_train)

# Make predictions for both training and test data using the trained Decision Tree
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

# Calculate accuracy for the default Decision Tree model on training and test sets
train_accuracy_dt = accuracy_score(y_train, y_train_pred_dt)
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)

# Output training and test accuracy for the default Decision Tree model
print(f"Default Decision Tree - Training Accuracy: {train_accuracy_dt:.2f}")
print(f"Default Decision Tree - Test Accuracy: {test_accuracy_dt:.2f}")

# Hyperparameter tuning for the Decision Tree (max_depth and min_samples_split)
dt_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)

# Train the tuned Decision Tree model
dt_tuned.fit(X_train, y_train)

# Make predictions for training and test sets using the tuned Decision Tree
y_train_pred_dt_tuned = dt_tuned.predict(X_train)
y_test_pred_dt_tuned = dt_tuned.predict(X_test)

# Calculate accuracy for the tuned Decision Tree model on both datasets
train_accuracy_dt_tuned = accuracy_score(y_train, y_train_pred_dt_tuned)
test_accuracy_dt_tuned = accuracy_score(y_test, y_test_pred_dt_tuned)

# Output training and test accuracy for the tuned Decision Tree model
print(f"Tuned Decision Tree - Training Accuracy: {train_accuracy_dt_tuned:.2f}")
print(f"Tuned Decision Tree - Test Accuracy: {test_accuracy_dt_tuned:.2f}")

# RANDOM FOREST MODEL

# Initialize Random Forest classifier with default hyperparameters
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest model on the training data
rf_model.fit(X_train, y_train)

# Make predictions for both training and test sets with the Random Forest model
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Compute accuracy for the default Random Forest model on both datasets
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)

# Output training and test accuracy for the default Random Forest model
print(f"Default Random Forest - Training Accuracy: {train_accuracy_rf:.2f}")
print(f"Default Random Forest - Test Accuracy: {test_accuracy_rf:.2f}")

# Hyperparameter tuning for Random Forest (n_estimators and max_depth)
rf_tuned = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the tuned Random Forest model on the training data
rf_tuned.fit(X_train, y_train)

# Make predictions for training and test sets using the tuned Random Forest
y_train_pred_rf_tuned = rf_tuned.predict(X_train)
y_test_pred_rf_tuned = rf_tuned.predict(X_test)

# Calculate accuracy for the tuned Random Forest model on both datasets
train_accuracy_rf_tuned = accuracy_score(y_train, y_train_pred_rf_tuned)
test_accuracy_rf_tuned = accuracy_score(y_test, y_test_pred_rf_tuned)

# Output training and test accuracy for the tuned Random Forest model
print(f"Tuned Random Forest - Training Accuracy: {train_accuracy_rf_tuned:.2f}")
print(f"Tuned Random Forest - Test Accuracy: {test_accuracy_rf_tuned:.2f}")

# RESULTS COMPARISON

# Output summary of results for both default and tuned models
print("\nSummary of Results\n")
print(f"Default Decision Tree - Train: {train_accuracy_dt:.2f}, Test: {test_accuracy_dt:.2f}")
print(f"Tuned Decision Tree   - Train: {train_accuracy_dt_tuned:.2f}, Test: {test_accuracy_dt_tuned:.2f}")
print(f"Default Random Forest - Train: {train_accuracy_rf:.2f}, Test: {test_accuracy_rf:.2f}")
print(f"Tuned Random Forest   - Train: {train_accuracy_rf_tuned:.2f}, Test: {test_accuracy_rf_tuned:.2f}")

# Prepare data for plotting model accuracy comparison
models = ['Default DT', 'Tuned DT', 'Default RF', 'Tuned RF']
train_accuracies = [train_accuracy_dt, train_accuracy_dt_tuned, train_accuracy_rf, train_accuracy_rf_tuned]
test_accuracies = [test_accuracy_dt, test_accuracy_dt_tuned, test_accuracy_rf, test_accuracy_rf_tuned]

# Create a DataFrame for accuracy results to facilitate plotting
accuracy_df = pd.DataFrame({
    'Model': models * 2,
    'Accuracy': train_accuracies + test_accuracies,
    'Dataset': ['Train'] * 4 + ['Test'] * 4
})

# Plot model accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=accuracy_df)
plt.ylim(0, 1)  # Accuracy range between 0 and 1
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Model Type')
plt.legend(loc='lower right')
plt.show()

# DECISION TREE VISUALIZATION

# Re-train a decision tree for visualization (adjust depth for clarity)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Calculate accuracy on training data for this visualization model
predictions_training = model.predict(X_train)
accuracy_training = accuracy_score(y_train, predictions_training)
print("Accuracy on training data:", accuracy_training)

# Visualize the decision tree structure
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X_train.columns, class_names=['No Fire', 'Fire'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# IMPACT OF HYPERPARAMETERS ON MODEL PERFORMANCE

# Define hyperparameter values to explore for Decision Tree
max_depth_values = [3, 5, 7, 10, 15]
min_samples_split_values = [2, 5, 10, 20]

# Store hyperparameter tuning results for Decision Tree
dt_results = []

# Loop over hyperparameter combinations for Decision Tree, train, and evaluate each
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        dt_model.fit(X_train, y_train)
        y_test_pred = dt_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Output current hyperparameter configuration and resulting accuracy
        print(f"Decision Tree - max_depth: {max_depth}, min_samples_split: {min_samples_split}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        dt_results.append({
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'test_accuracy': test_accuracy
        })

# Convert Decision Tree hyperparameter tuning results into DataFrame
dt_results_df = pd.DataFrame(dt_results)

# Plot impact of Decision Tree hyperparameters on test accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(x='max_depth', y='test_accuracy', hue='min_samples_split', marker="o", data=dt_results_df)
plt.title('Decision Tree: Impact of Hyperparameters on Test Accuracy')
plt.ylabel('Test Accuracy')
plt.xlabel('Max Depth')
plt.show()

# RANDOM FOREST HYPERPARAMETER TUNING

# Define hyperparameter values to explore for Random Forest
n_estimators_values = [50, 100, 150, 200]
max_depth_values_rf = [5, 10, 15, 20]

# Store hyperparameter tuning results for Random Forest
rf_results = []

# Loop over hyperparameter combinations for Random Forest, train, and evaluate each
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values_rf:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        y_test_pred = rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Output current hyperparameter configuration and resulting accuracy
        print(f"Random Forest - n_estimators: {n_estimators}, max_depth: {max_depth}")
        print(f"Test Accuracy: {test_accuracy:.2f}")

        rf_results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_accuracy': test_accuracy
        })

# Convert Random Forest hyperparameter tuning results into DataFrame
rf_results_df = pd.DataFrame(rf_results)

# Plot impact of Random Forest hyperparameters on test accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(x='n_estimators', y='test_accuracy', hue='max_depth', marker="o", data=rf_results_df)
plt.title('Random Forest: Impact of Hyperparameters on Test Accuracy')
plt.ylabel('Test Accuracy')
plt.xlabel('Number of Estimators')
plt.show()
