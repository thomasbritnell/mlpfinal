import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import time
import pickle
import os
from datetime import datetime

# Load and preprocess both datasets separately
def load_data():
    """Load and preprocess both datasets separately"""
    print("Loading datasets...")
    
    # Dataset 1 - needs label mapping
    train_df1 = pd.read_csv('datasets/fake_or_real_news_train_feature.csv')
    test_df1 = pd.read_csv('datasets/fake_or_real_news_test_feature.csv')
    
    # Map text labels to binary values for dataset 1
    label_map = {'FAKE': 0, 'REAL': 1}
    train_df1['label'] = train_df1['label'].map(label_map)
    test_df1['label'] = test_df1['label'].map(label_map)
    
    # Dataset 2 - already has binary labels
    train_df2 = pd.read_csv('datasets/fulltrain_Guardian_Nyt_binary_shuffled_feature.csv')
    test_df2 = pd.read_csv('datasets/Mixed_and_fulltrain_feature.csv')
    
    # Check if labels in Dataset 2 need to be converted to integers
    if train_df2['label'].dtype != 'int64':
        train_df2['label'] = train_df2['label'].astype(int)
    if test_df2['label'].dtype != 'int64':
        test_df2['label'] = test_df2['label'].astype(int)
    
    # Check class distribution for both datasets
    print("FAKE OR REAL Dataset:")
    train_class_dist1 = dict(train_df1['label'].value_counts())
    test_class_dist1 = dict(test_df1['label'].value_counts())
    print(f"Class distribution - Train: {train_class_dist1}")
    print(f"Class distribution - Test: {test_class_dist1}")
    
    print("\nCOMBINED CORPUS Dataset:")
    train_class_dist2 = dict(train_df2['label'].value_counts())
    test_class_dist2 = dict(test_df2['label'].value_counts())
    print(f"Class distribution - Train: {train_class_dist2}")
    print(f"Class distribution - Test: {test_class_dist2}")
    
    return (train_df1, test_df1), (train_df2, test_df2)

# Apply SMOTE for handling class imbalance
def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to handle class imbalance"""
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    return X_train_resampled, y_train_resampled

# GridSearchCV for hyperparameter tuning
def tune_random_forest_grid(X_train, y_train, cv=5, dataset_name=""):
    """Tune Random Forest hyperparameters using GridSearchCV"""
    print(f"Starting GridSearchCV hyperparameter tuning for Random Forest on {dataset_name}...")
    start_time = time.time()
    
    # Create parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        scoring='f1'
    )
    
    # Fit the grid search model
    grid_search.fit(X_train, y_train)
    
    # Duration of tuning
    duration = time.time() - start_time
    print(f"Grid search completed in {duration:.2f} seconds")
    
    # Best parameters and score
    print(f"\nBest Parameters for {dataset_name}:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Evaluate model function
def evaluate_model(model, X_test, y_test, dataset_name):
    """Evaluate model performance with various metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Print results
    print(f"\nModel Performance on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")
    
    # Classification report
    print(f"Classification Report for {dataset_name}:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print(f"Confusion Matrix for {dataset_name}:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

# Function to analyze feature importance
def analyze_features(model, feature_names, dataset_name):
    """Analyze and print feature importance"""
    feature_importances = model.feature_importances_
    
    # Sort feature importances
    sorted_indices = np.argsort(feature_importances)[::-1]
    
    print(f"\nFeature Importance Ranking for {dataset_name}:")
    for i in range(len(feature_names)):
        idx = sorted_indices[i]
        print(f"{feature_names[idx]} - {feature_importances[idx]:.4f}")
        
    # Return feature importance as a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df

# Create feature engineering function to be applied to all datasets
def engineer_features(df):
    """Apply feature engineering to the dataset"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Sentiment ratios
    df_copy['neg_pos_ratio'] = df_copy['sent_neg'] / (df_copy['sent_pos'] + 0.001)
    
    # Adjective density
    df_copy['adj_density'] = df_copy['adjectives'] / (df_copy['WordCount'] + 0.001)
    
    # Word length to article length ratio
    df_copy['word_article_ratio'] = df_copy['WordCount'] / (df_copy['Article_len'] + 0.001)
    
    return df_copy

# Function to save model and related information using pickle
def save_model(model, feature_columns, metrics, dataset_name):
    """Save model, feature list, and metrics using pickle"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a model info dictionary
    model_info = {
        'model': model,
        'feature_columns': feature_columns,
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    # Save the model info
    filename = f"models/random_forest_{dataset_name.replace(' ', '_').lower()}_{timestamp}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model_info, file)
    
    print(f"\nModel for {dataset_name} saved to {filename}")
    return filename

# Function to load a saved model
def load_model(filename):
    """Load a saved model and its information"""
    with open(filename, 'rb') as file:
        model_info = pickle.load(file)
    
    print(f"Model loaded from {filename}")
    print(f"Model timestamp: {model_info['timestamp']}")
    
    return model_info

# Function to process a single dataset
def process_dataset(dataset_tuple, dataset_name):
    train_df, test_df = dataset_tuple
    
    # Select features
    feature_columns = [
        'Article_len', 'Avg_Word_Len', 'CountOfNumbers', 'CountofExclamation',
        'adjectives', 'WordCount', 'sent_neg', 'sent_neu', 'sent_pos'
    ]
    
    # Create additional features
    print(f"\nCreating additional features for {dataset_name}...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Add the engineered features to our feature list
    engineered_features = ['neg_pos_ratio', 'adj_density', 'word_article_ratio']
    feature_columns += engineered_features
    
    # Prepare training data
    X_train = train_df[feature_columns].values
    y_train = train_df['label'].values
    
    # Prepare test data
    X_test = test_df[feature_columns].values
    y_test = test_df['label'].values
    
    # Apply SMOTE to handle class imbalance
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Tune with GridSearchCV
    best_rf_model = tune_random_forest_grid(
        X_train_resampled, y_train_resampled, cv=5, dataset_name=dataset_name
    )
    
    # Evaluate the tuned model
    print(f"\nEvaluating tuned model for {dataset_name}...")
    results = evaluate_model(best_rf_model, X_test, y_test, dataset_name)
    
    # Analyze feature importance
    importance_df = analyze_features(best_rf_model, feature_columns, dataset_name)
    
    # Save the model
    saved_model_path = save_model(best_rf_model, feature_columns, results, dataset_name)
    
    return results, importance_df, best_rf_model, saved_model_path

# Main execution function
def main():
    # Load the datasets
    dataset1, dataset2 = load_data()
    
    # Process Dataset 1
    print("\n" + "="*50)
    print("PROCESSING FAKE OR REAL Dataset")
    print("="*50)
    results1, importance_df1, model1, model1_path = process_dataset(dataset1, "FAKE OR REAL Dataset")
    
    # Process Dataset 2
    print("\n" + "="*50)
    print("PROCESSING COMBINED CORPUS Dataset")
    print("="*50)
    results2, importance_df2, model2, model2_path = process_dataset(dataset2, "COMBINED CORPUS Dataset")
    
    # Print a summary of performance for both datasets
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"FAKE OR REAL Dataset - F1 Score: {results1['f1']:.4f}, Accuracy: {results1['accuracy']:.4f}")
    print(f"COMBINED CORPUS Dataset - F1 Score: {results2['f1']:.4f}, Accuracy: {results2['accuracy']:.4f}")
    
    # Compare top features between datasets
    print("\n" + "="*50)
    print("TOP FEATURES COMPARISON")
    print("="*50)
    top_features1 = importance_df1.head(5)['Feature'].tolist()
    top_features2 = importance_df2.head(5)['Feature'].tolist()
    
    print(f"Top 5 features for FAKE OR REAL Dataset: {', '.join(top_features1)}")
    print(f"Top 5 features for COMBINED CORPUS Dataset: {', '.join(top_features2)}")
    
    # Find common important features
    common_features = set(top_features1).intersection(set(top_features2))
    print(f"\nCommon important features: {', '.join(common_features) if common_features else 'None'}")
    
    # Print saved model paths
    print("\n" + "="*50)
    print("SAVED MODELS")
    print("="*50)
    print(f"FAKE OR REAL Dataset model saved to: {model1_path}")
    print(f"COMBINED CORPUS Dataset model saved to: {model2_path}")
    
    # Example of loading a saved model (commented out by default)
    print("\n" + "="*50)
    print("MODEL LOADING EXAMPLE")
    print("="*50)
    print("To load the saved models in another script, use:")
    print(f"loaded_model_info = load_model('{model1_path}')")
    print("model = loaded_model_info['model']")
    print("feature_columns = loaded_model_info['feature_columns']")
    print("metrics = loaded_model_info['metrics']")

# Function to make predictions with a saved model
def predict_with_saved_model(filename, new_data):
    """Make predictions using a saved model"""
    # Load the model
    model_info = load_model(filename)
    model = model_info['model']
    feature_columns = model_info['feature_columns']
    
    # Ensure new_data has the required features
    if isinstance(new_data, pd.DataFrame):
        # Engineer features if needed
        new_data = engineer_features(new_data)
        # Extract the required features
        X = new_data[feature_columns].values
    else:
        # Assume new_data is already the right format
        X = new_data
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities

if __name__ == "__main__":
    main()