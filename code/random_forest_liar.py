import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data(train_path, test_path):
    
    print("Loading datasets...")
    train_df = pd.read_csv('datasets/train.csv')
    test_df = pd.read_csv('datasets/test.csv')
    
    # Map text labels to binary values if needed
    if train_df['Label'].dtype == object:
        label_map = {'False': 0, 'True': 1, False: 0, True: 1}
        train_df['Label'] = train_df['Label'].map(label_map)
        test_df['Label'] = test_df['Label'].map(label_map)
    
    # Check class distribution
    train_class_dist = dict(train_df['Label'].value_counts())
    test_class_dist = dict(test_df['Label'].value_counts())
    
    print(f"Class distribution - Train: {train_class_dist}")
    print(f"Class distribution - Test: {test_class_dist}")
    
    return train_df, test_df

# Feature engineering 
def engineer_features(df):
    """Extract statistical and linguistic features from text"""
    print("Engineering features...")
    
    features = pd.DataFrame()
    
    # Basic counts
    features['char_count'] = df['Statement'].apply(len).astype(np.float64)
    features['word_count'] = df['Statement'].apply(lambda x: len(str(x).split())).astype(np.float64)
    features['unique_word_count'] = df['Statement'].apply(lambda x: len(set(str(x).lower().split()))).astype(np.float64)
    features['unique_word_ratio'] = (features['unique_word_count'] / (features['word_count'] + 1)).astype(np.float64)
    
    # Average word length
    features['avg_word_length'] = df['Statement'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    ).astype(np.float64)
    
    # Sentence features
    import re
    features['sentence_count'] = df['Statement'].apply(lambda x: len(re.split(r'[.!?]+', str(x)))).astype(np.float64)
    features['avg_sentence_length'] = (features['word_count'] / (features['sentence_count'] + 1)).astype(np.float64)
    
    # Special character counts
    features['exclamation_count'] = df['Statement'].apply(lambda x: str(x).count('!')).astype(np.float64)
    features['question_count'] = df['Statement'].apply(lambda x: str(x).count('?')).astype(np.float64)
    features['capital_count'] = df['Statement'].apply(lambda x: sum(1 for c in str(x) if c.isupper())).astype(np.float64)
    features['capital_ratio'] = (features['capital_count'] / (features['char_count'] + 1)).astype(np.float64)
    
    # Convert boolean flags to float for sparse compatibility
    features['has_number'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\d', str(x)))))
    features['has_quote'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\"|\"|\'|\'', str(x)))))
    features['has_source'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\bsource\b|\baccording\b', str(x).lower()))))
    
    # Make sure all features are numeric and not object type
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].astype(np.float64)
    
    return features

# Text vectorization function
def vectorize_text(train_df, test_df, max_features=2000):
    """Vectorize text using Count Vectorizer"""
    print("Vectorizing text...")
    
    vectorizer = CountVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5
    )
    
    # Fit and transform
    X_text_train = vectorizer.fit_transform(train_df['Statement'])
    X_text_test = vectorizer.transform(test_df['Statement'])
    
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    return X_text_train, X_text_test, vectorizer

# Combine features function
def combine_features(X_text_train, X_text_test, X_stats_train, X_stats_test):
    """Combine text vectors and statistical features"""
    # Convert DataFrame to CSR matrix for hstack
    X_stats_train_csr = csr_matrix(X_stats_train.astype(np.float64).values)
    X_stats_test_csr = csr_matrix(X_stats_test.astype(np.float64).values)
    
    # Combine features using hstack
    X_train_combined = hstack([X_text_train, X_stats_train_csr])
    X_test_combined = hstack([X_text_test, X_stats_test_csr])
    
    print(f"Final feature shape: {X_train_combined.shape}")
    
    # Convert to dense array for algorithms that don't work with sparse matrices
    X_train_combined = X_train_combined.toarray()
    X_test_combined = X_test_combined.toarray()
    
    return X_train_combined, X_test_combined

# Apply SMOTE for handling class imbalance
def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to handle class imbalance"""
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    return X_train_resampled, y_train_resampled

# Random Forest model
def train_random_forest(X_train, y_train):
    """Train a Random Forest model"""
    print("Training Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=1,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    return rf_model

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
    
    print("\nTop 20 Important Features:")
    for i in range(min(20, len(feature_names))):
        idx = sorted_indices[i]
        print(f"{feature_names[idx]} - {feature_importances[idx]:.4f}")

# Main execution function
def main():
    # Paths to data
    train_path = "datasets/train.csv"  
    test_path = "datasets/train.csv"    
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    
    # Engineer features
    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)
    
    # Vectorize text
    X_text_train, X_text_test, vectorizer = vectorize_text(train_df, test_df, max_features=2000)
    
    # Combine features
    X_train, X_test = combine_features(X_text_train, X_text_test, train_features, test_features)
    y_train = train_df['Label'].values
    y_test = test_df['Label'].values
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    results = evaluate_model(rf_model, X_test, y_test)
    
    # Analyze feature importance
    # Combine feature names from vectorizer and engineered features
    all_feature_names = list(vectorizer.get_feature_names_out()) + list(train_features.columns)
    analyze_features(rf_model, all_feature_names)
    
    # Save the model
    import pickle
    with open('fake_news_detector_rf_liar.pkl', 'wb') as model_file:
        pickle.dump({'model': rf_model, 'vectorizer': vectorizer}, model_file)
    
    print("\nModel saved as 'fake_news_detector_rf_lier.pkl'")

if __name__ == "__main__":
    main()