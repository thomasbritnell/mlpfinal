#Note : this saves "fake_news_detector_rf_liar.pickle" locally


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data():
    
    train_df = pd.read_csv('../datasets/train.csv')
    test_df = pd.read_csv('../datasets/test.csv')
    
    #encode the target 
    label_map = {False: 0, True: 1}
    train_df['Label'] = train_df['Label'].map(label_map)
    test_df['Label'] = test_df['Label'].map(label_map)
    print(f"Data Sample:\n{train_df.head()}\n")
    return train_df, test_df

# Feature engineering 
def engineer_features(df):
    
    
    features = pd.DataFrame()
    
    # Counts of characters, words, unique words, etc
    features['char_count'] = df['Statement'].apply(len).astype(np.float64)
    features['word_count'] = df['Statement'].apply(lambda x: len(str(x).split())).astype(np.float64)
    features['unique_word_count'] = df['Statement'].apply(lambda x: len(set(str(x).lower().split()))).astype(np.float64)
    # this is a calculation of how many words are unique in the sample 
    features['unique_word_ratio'] = (features['unique_word_count'] / (features['word_count'] + 1)).astype(np.float64)
    
    #average word length
    features['avg_word_length'] = df['Statement'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    ).astype(np.float64)
    
    #  Sentence features using regular expressions
    import re
    features['sentence_count'] = df['Statement'].apply(lambda x: len(re.split(r'[.!?]+', str(x)))).astype(np.float64)
    features['avg_sentence_length'] = (features['word_count'] / (features['sentence_count'] + 1)).astype(np.float64)
    
    # Special characters like exclamation (might indicate a sensational article title for example)
    features['exclamation_count'] = df['Statement'].apply(lambda x: str(x).count('!')).astype(np.float64)
    features['question_count'] = df['Statement'].apply(lambda x: str(x).count('?')).astype(np.float64)
    features['capital_count'] = df['Statement'].apply(lambda x: sum(1 for c in str(x) if c.isupper())).astype(np.float64)
    features['capital_ratio'] = (features['capital_count'] / (features['char_count'] + 1)).astype(np.float64)
    
    # flags for different things like the inclusion of quotes, or if there is a source cited 
    features['has_number'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\d', str(x)))))
    features['has_quote'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\"|\"|\'|\'', str(x)))))
    features['has_source'] = df['Statement'].apply(lambda x: float(bool(re.search(r'\bsource\b|\baccording\b', str(x).lower()))))

    
    return features

# Text vectorization function with sklearn CountVectorizer
def vectorize_text(train_df, test_df, max_features=2000):
   
    #use bigram and unigram
    vectorizer = CountVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5
    )
    
    # Fit and transform
    X_text_train = vectorizer.fit_transform(train_df['Statement'])
    X_text_test = vectorizer.transform(test_df['Statement'])
    
    return X_text_train, X_text_test, vectorizer

# Combine the vectorized text with the added derived features
def combine_features(X_text_train, X_text_test, X_stats_train, X_stats_test):
   
    # csr to compress the spare matrix from vectorization
    X_stats_train_csr = csr_matrix(X_stats_train.astype(np.float64).values)
    X_stats_test_csr = csr_matrix(X_stats_test.astype(np.float64).values)
    
    # combine all the engineered features and the vectorized 
    X_train_combined = hstack([X_text_train, X_stats_train_csr])
    X_test_combined = hstack([X_text_test, X_stats_test_csr])

    
    return X_train_combined, X_test_combined

# smote for handling class imbalance
def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    return X_train_resampled, y_train_resampled

# Random Forest classifier
def train_random_forest(X_train, y_train):
    
    #params were chosen with GridSearchCV, though not incldued because of the long run-time
    rf_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=1,
        max_depth=None,
        class_weight='balanced',
        random_state=22,
        n_jobs=-1 
    )
    
    rf_model.fit(X_train, y_train)
    
    return rf_model

# Evaluation
def evaluate_model(model, X_test, y_test):

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] #get probabilities for both true and false, needed for roc
    
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
    
# feature importance
def analyze_features(model, feature_names):
     """Analyze and print feature importance"""    """Analyze and print feature importance"""
     feature_importances = model.feature_importances_
     
     # Sort feature importances
     sorted_indices = np.argsort(feature_importances)[::-1]
     
     print("\nTop 10 Important Features:")
     for i in range(min(10, len(feature_names))):
         idx = sorted_indices[i]
         print(f"{feature_names[idx]} - {feature_importances[idx]:.4f}")

def main():  
    
    # Load and preprocess data
    train_df, test_df = load_data()
    
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
    
    all_feature_names = list(vectorizer.get_feature_names_out()) + list(train_features.columns)
    analyze_features(rf_model, all_feature_names) 
    
    # Save the model
    import pickle
    with open('fake_news_detector_rf_liar.pickle', 'wb') as model_file:
        pickle.dump({'model': rf_model, 'vectorizer': vectorizer}, model_file)
    
    print("\nModel saved as 'fake_news_detector_rf_liar.pickle'")

if __name__ == "__main__":
    main()