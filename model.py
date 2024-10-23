import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib  # For saving the model

# Load your entire dataset
data = pd.read_csv('/content/drive/MyDrive/project/data.csv')  # Full dataset
X_full = data.drop(columns=['Label'])  # Features
y_full = data['Label']  # Target variable

# Train-Test Split (for internal validation, optional)
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.3, random_state=42)

# Define classifiers
classifiers = {
    'SGD': SGDClassifier(),
    'RandomForest': RandomForestClassifier(),
    'MLP': MLPClassifier(),
    'SVM': SVC()
}

# Store results for internal validation
cv_results = []

for name, clf in classifiers.items():
    # Cross-validation on training data
    cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
    mean_cv_accuracy = np.mean(cv_scores)
    
    # Train the model on the full training data
    clf.fit(X_train, y_train)

    # Save the model
    joblib.dump(clf, f'/content/drive/MyDrive/project/result_act/{name}_model.pkl')

    # Optionally evaluate on validation data (if you want)
    val_accuracy = clf.score(X_val, y_val)

    # Store cross-validation results
    cv_results.append({
        'Model': name,
        'CV Mean Accuracy': mean_cv_accuracy,
        'Validation Accuracy': val_accuracy
    })

# Convert CV results to DataFrame
cv_results_df = pd.DataFrame(cv_results)

# Now use a separate test set for final evaluation
test_data = pd.read_csv('/content/drive/MyDrive/project/test.csv')  # Separate test dataset
X_test = test_data.drop(columns=['Label'])  # Features
y_test = test_data['Label']  # Target variable

# Test the models on the separate test data
test_results = []

for name in classifiers.keys():
    clf = joblib.load(f'/content/drive/MyDrive/project/result_act/{name}_model.pkl')  # Load the model
    test_accuracy = clf.score(X_test, y_test)  # Evaluate on test set
    
    test_results.append({
        'Model': name,
        'Test Accuracy': test_accuracy
    })

# Convert test results to DataFrame
test_results_df = pd.DataFrame(test_results)

# Save the results to CSV files
cv_results_df.to_csv('/content/drive/MyDrive/project/result_act/cross_validation_results.csv', index=False)
test_results_df.to_csv('/content/drive/MyDrive/project/result_act/test_results.csv', index=False)

# Print results
print("Cross-Validation Results:")
print(cv_results_df)
print("\nTest Results:")
print(test_results_df)
