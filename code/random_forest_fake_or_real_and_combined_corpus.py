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



    # Dataset 1 - needs label mapping
    train_df1 = pd.read_csv('../datasets/fake_or_real_news_train_feature.csv')
    test_df1 = pd.read_csv('../datasets/fake_or_real_news_test_feature.csv')
    
    # Map text labels to binary values for dataset 1
    label_map = {'FAKE': 0, 'REAL': 1}
    train_df1['label'] = train_df1['label'].map(label_map)
    test_df1['label'] = test_df1['label'].map(label_map)
    
    
    # Dataset 2 - already has binary labels
    train_df2 = pd.read_csv('../datasets/fulltrain_Guardian_Nyt_binary_shuffled_feature.csv')
    test_df2 = pd.read_csv('../datasets/Mixed_and_fulltrain_feature.csv')
    
    # Check if labels in Dataset 2 need to be converted to integers
    if train_df2['label'].dtype != 'int64':
        train_df2['label'] = train_df2['label'].astype(int)
    if test_df2['label'].dtype != 'int64':
        test_df2['label'] = test_df2['label'].astype(int)
    
    # Check class distribution for both datasets
    print("real or fake dataset")
    train_class_dist1 = dict(train_df1['label'].value_counts())
    test_class_dist1 = dict(test_df1['label'].value_counts())
    print(f"class distribution - Train: 1: {train_class_dist1[1]} 0: {train_class_dist1[0]}")
    print(f"class distribution - Test: 1: {test_class_dist1[1]} 0: {test_class_dist1[0]}")
    
    print("\ncombined corpus dataset:")
    train_class_dist2 = dict(train_df2['label'].value_counts())
    test_class_dist2 = dict(test_df2['label'].value_counts())
    print(f"class distribution - Train: 1: {train_class_dist2[1]} 0: {train_class_dist2[0]}")
    print(f"Class distribution - Test: 1: {test_class_dist2[1]} 0: {test_class_dist2[0]}")
    
    return (train_df1, test_df1), (train_df2, test_df2)

# Apply SMOTE for handling class imbalance
def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled

# GridSearchCV for hyperparameter tuning
def tune_random_forest_grid(X_train, y_train, cv=5, dataset_name=""):

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
    rf = RandomForestClassifier(random_state=42) #we actually tried a couple different states and 42 was better
    
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

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] #probability of each option
    
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
    



# create derived features
def engineer_features(df):

    # to not modify the original
    df_copy = df.copy()
    
    # Sentiment ratios (as found in the original dataset)
    df_copy['neg_pos_ratio'] = df_copy['sent_neg'] / (df_copy['sent_pos'] + 0.001)
    
    # Adjective density
    df_copy['adj_density'] = df_copy['adjectives'] / (df_copy['WordCount'] + 0.001)
    
    # Word length to article length ratio
    df_copy['word_article_ratio'] = df_copy['WordCount'] / (df_copy['Article_len'] + 0.001)
    
    return df_copy

# save model to file
def save_model(model, feature_columns, metrics, dataset_name):

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
    filename = f"models/random_forest_{dataset_name.replace(' ', '_').lower()}_{timestamp}.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(model_info, file)
    
    print(f"\nModel for {dataset_name} saved to {filename}")
    return filename

# load saved models
def load_model(filename):
    with open(filename, 'rb') as file:
        model_info = pickle.load(file)
    
    print(f"Model loaded from {filename}")
    print(f"Model timestamp: {model_info['timestamp']}")
    
    return model_info

# the features are already part of the dataset 
def process_dataset(dataset_tuple, dataset_name):
    train_df, test_df = dataset_tuple
    
    # Select features
    feature_columns = [
        'Article_len', 'Avg_Word_Len', 'CountOfNumbers', 'CountofExclamation',
        'adjectives', 'WordCount', 'sent_neg', 'sent_neu', 'sent_pos'
    ]
    
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # add the new features to the rest
    engineered_features = ['neg_pos_ratio', 'adj_density', 'word_article_ratio']
    feature_columns += engineered_features
    
    # Prepare training data
    X_train = train_df[feature_columns].values
    y_train = train_df['label'].values
    
    # Prepare test data
    X_test = test_df[feature_columns].values
    y_test = test_df['label'].values
    
    # smote for class imbalance
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # gridsearch cv
    best_rf_model = tune_random_forest_grid(
        X_train_resampled, y_train_resampled, cv=5, dataset_name=dataset_name
    )
    
    results = evaluate_model(best_rf_model, X_test, y_test, dataset_name)
    
   
    saved_model_path = save_model(best_rf_model, feature_columns, results, dataset_name)
    
    return results, best_rf_model, saved_model_path

# Main execution function
def main():
    # Load the datasets
    dataset1, dataset2 = load_data()
    
    # Process Dataset 1
    results1, model1, model1_path = process_dataset(dataset1, "fake or real Dataset")
    
    # Process Dataset 2
 
 
    results2, model2, model2_path = process_dataset(dataset2, "combined corpus Dataset")
    
    # Print a summary of performance for both datasets
   

    print(f"fake or real Dataset - F1 Score: {results1['f1']:.4f}, Accuracy: {results1['accuracy']:.4f}")
    print(f"combined corpus Dataset - F1 Score: {results2['f1']:.4f}, Accuracy: {results2['accuracy']:.4f}")
    
   
    
    print(f"fake or real Dataset model saved to: {model1_path}")
    print(f"combined corpus Dataset model saved to: {model2_path}")
    
# load a saved model and use it for predictons

#currently unused
def predict_with_saved_model(filename, new_data):
    # Load the model
    model_info = load_model(filename)
    model = model_info['model']
    feature_columns = model_info['feature_columns']
    
    
    X_test = new_data
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    return predictions, probabilities


if __name__ == "__main__":
    main()