import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import time
from scipy.stats import randint, uniform

# Load and preprocess data
def load_data(train_path, test_path):
    """Load and preprocess the dataset with the new structure"""
    print("Loading datasets...")
    train_df = pd.read_csv('FakeNewsDetection/Codes/Traditional_ML_based_Methods/1_SVM_Lexical/1_liar__lexical/liar_train_feature.csv')
    test_df = pd.read_csv('FakeNewsDetection/Codes/Traditional_ML_based_Methods/2_SVM__LexicalSentiment/1_liar__lex_sent/liar_test_feature.csv')
    
    # Map text labels to binary values
    label_map = {'False': 0, 'True': 1, False: 0, True: 1}
    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)
    
    # Check class distribution
    train_class_dist = dict(train_df['label'].value_counts())
    test_class_dist = dict(test_df['label'].value_counts())
    
    print(f"Class distribution - Train: {train_class_dist}")
    print(f"Class distribution - Test: {test_class_dist}")
    
    return train_df, test_df

# Apply SMOTE for handling class imbalance
def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to handle class imbalance"""
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    return X_train_resampled, y_train_resampled

# New function for hyperparameter tuning with RandomizedSearchCV
def tune_random_forest(X_train, y_train, cv=5):
    """Tune Random Forest hyperparameters using RandomizedSearchCV"""
    print("Starting hyperparameter tuning for Random Forest...")
    start_time = time.time()
    
    # Parameter distributions for RandomizedSearchCV
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(randint(10, 100).rvs(5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None] + list(uniform(0.1, 0.9).rvs(3)),
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # RandomizedSearchCV
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings to try
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='f1'  # You can change to accuracy, precision, recall as needed
    )
    
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    # Duration of tuning
    duration = time.time() - start_time
    print(f"Hyperparameter tuning completed in {duration:.2f} seconds")
    
    # Best parameters and score
    print("\nBest Parameters:")
    for param, value in rf_random.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best F1 Score: {rf_random.best_score_:.4f}")
    
    # Return the best model
    return rf_random.best_estimator_

# Alternative: GridSearchCV for more precise but slower tuning
def tune_random_forest_grid(X_train, y_train, cv=5):
    """Tune Random Forest hyperparameters using GridSearchCV"""
    print("Starting GridSearchCV hyperparameter tuning for Random Forest...")
    start_time = time.time()
    
    # Create parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
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
    print("\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Evaluate model function
def evaluate_model(model, X_test, y_test):
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
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
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
def analyze_features(model, feature_names):
    """Analyze and print feature importance"""
    feature_importances = model.feature_importances_
    
    # Sort feature importances
    sorted_indices = np.argsort(feature_importances)[::-1]
    
    print("\nFeature Importance Ranking:")
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

# Main execution function
def main():
    # Paths to your data
    train_path = "train_data.csv"  # Update with your actual path
    test_path = "test_data.csv"    # Update with your actual path
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    
    # Select features and target
    feature_columns = [
        'Article_len', 'Avg_Word_Len', 'CountOfNumbers', 'CountofExclamation',
        'adjectives', 'WordCount', 'sent_neg', 'sent_neu', 'sent_pos'
    ]
    
    # Create some additional features
    print("Creating additional features...")
    
    # Sentiment ratios
    train_df['neg_pos_ratio'] = train_df['sent_neg'] / (train_df['sent_pos'] + 0.001)
    test_df['neg_pos_ratio'] = test_df['sent_neg'] / (test_df['sent_pos'] + 0.001)
    
    # Adjective density
    train_df['adj_density'] = train_df['adjectives'] / (train_df['WordCount'] + 0.001)
    test_df['adj_density'] = test_df['adjectives'] / (test_df['WordCount'] + 0.001)
    
    # Word length to article length ratio
    train_df['word_article_ratio'] = train_df['WordCount'] / (train_df['Article_len'] + 0.001)
    test_df['word_article_ratio'] = test_df['WordCount'] / (test_df['Article_len'] + 0.001)
    
    # Update feature columns with new features
    feature_columns += ['neg_pos_ratio', 'adj_density', 'word_article_ratio']
    
    X_train = train_df[feature_columns].values
    y_train = train_df['label'].values
    
    X_test = test_df[feature_columns].values
    y_test = test_df['label'].values
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Choice between RandomizedSearchCV (faster) or GridSearchCV (more exhaustive)
    use_random_search = True  # Set to False to use GridSearchCV instead
    
    if use_random_search:
        # Tune with RandomizedSearchCV
        best_rf_model = tune_random_forest(X_train_resampled, y_train_resampled, cv=5)
    else:
        # Tune with GridSearchCV
        best_rf_model = tune_random_forest_grid(X_train_resampled, y_train_resampled, cv=5)
    
    # Evaluate the tuned model
    print("\nEvaluating tuned model on test data...")
    results = evaluate_model(best_rf_model, X_test, y_test)
    
    # Analyze feature importance
    importance_df = analyze_features(best_rf_model, feature_columns)
    
    # Save the model
    import pickle
    with open('fake_news_detector_tuned_rf.pkl', 'wb') as model_file:
        pickle.dump({
            'model': best_rf_model, 
            'features': feature_columns,
            'feature_importance': importance_df.to_dict()
        }, model_file)
    
    print("\nTuned model saved as 'fake_news_detector_tuned_rf.pkl'")

# Add function to load and test on new data
def test_on_new_data(model_path, data_path):
    """Test the model on a new dataset"""
    print(f"Testing model on new data: {data_path}")
    
    # Load the model
    import pickle
    with open(model_path, 'rb') as model_file:
        model_data = pickle.load(model_file)
    
    rf_model = model_data['model']
    feature_columns = model_data['features']
    
    # Load the new data
    test_data = pd.read_csv(data_path)
    
    # Create additional features if they were used during training
    if 'neg_pos_ratio' in feature_columns and 'neg_pos_ratio' not in test_data.columns:
        test_data['neg_pos_ratio'] = test_data['sent_neg'] / (test_data['sent_pos'] + 0.001)
    
    if 'adj_density' in feature_columns and 'adj_density' not in test_data.columns:
        test_data['adj_density'] = test_data['adjectives'] / (test_data['WordCount'] + 0.001)
    
    if 'word_article_ratio' in feature_columns and 'word_article_ratio' not in test_data.columns:
        test_data['word_article_ratio'] = test_data['WordCount'] / (test_data['Article_len'] + 0.001)
    
    # Map text labels to binary if needed
    if 'label' in test_data.columns and test_data['label'].dtype == object:
        label_map = {'FAKE': 0, 'REAL': 1}
        test_data['label'] = test_data['label'].map(label_map)
    
    # Select features
    X_test = test_data[feature_columns].values
    
    if 'label' in test_data.columns:
        y_test = test_data['label'].values
        
        # Evaluate the model
        results = evaluate_model(rf_model, X_test, y_test)
    else:
        # Just make predictions
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Map back to text labels
        label_map_reverse = {0: 'FAKE', 1: 'REAL'}
        predictions = [label_map_reverse[p] for p in y_pred]
        
        # Add predictions to the data
        test_data['predicted_label'] = predictions
        test_data['prediction_probability'] = y_proba
        
        # Save predictions
        test_data.to_csv('predictions.csv', index=False)
        print("Predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If command-line arguments are provided, use test_on_new_data
        if len(sys.argv) < 3:
            print("Usage: python script.py model_path data_path")
            sys.exit(1)
        
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        test_on_new_data(model_path, data_path)
    else:
        # Otherwise run the main training pipeline
        main()