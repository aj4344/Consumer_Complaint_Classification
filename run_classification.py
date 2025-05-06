import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model, metrics, svm

print("Loading and preparing data...")
# Load the data
try:
    Data = pd.read_csv('consumer_complaints.csv', encoding='latin-1')
    print(f"Data loaded successfully with {Data.shape[0]} rows and {Data.shape[1]} columns")
    
    # Keep only relevant columns and rows with non-null complaints
    Data = Data[['product', 'consumer_complaint_narrative']]
    Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]
    print(f"After filtering: {Data.shape[0]} rows with valid complaint narratives")
    
    # Show class distribution
    print("\nComplaint distribution by product category:")
    print(Data.groupby('product').consumer_complaint_narrative.count())
    
    print("\nProcessing text data with TF-IDF...")
    # TF-IDF Vectorization
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(Data['consumer_complaint_narrative'])
    
    # Split data
    train_x, valid_x, train_y, valid_y = train_test_split(
        Data['consumer_complaint_narrative'], 
        Data['product'], 
        test_size=0.25,
        random_state=42
    )
    
    print(f"Training set: {len(train_x)} samples, Validation set: {len(valid_x)} samples")
    
    # Encode labels
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y_encoded = encoder.transform(valid_y)
    
    # Transform text to TF-IDF features
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)
    
    print("\nTraining Logistic Regression model...")
    # Train Logistic Regression with class weights
    model_lr = linear_model.LogisticRegression(
        C=10, 
        class_weight={8: 3.0, 9: 3.0, 5: 3.0, 7: 20.0},
        max_iter=1000
    ).fit(xtrain_tfidf, train_y)
    
    print("\nEvaluating Logistic Regression model...")
    # Evaluate Logistic Regression
    lr_predictions = model_lr.predict(xvalid_tfidf)
    accuracy = metrics.accuracy_score(lr_predictions, valid_y_encoded)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(metrics.classification_report(valid_y_encoded, lr_predictions, target_names=Data['product'].unique()))
    
    # Save the logistic regression model
    print("\nSaving Logistic Regression model...")
    with open('customer_classification_model_lr.pkl', 'wb') as file:
        pickle.dump(model_lr, file)
    
    print("\nTraining SVM model...")
    # Train SVM with class weights
    model_svm = svm.LinearSVC(
        C=100, 
        class_weight={8: 3.0, 9: 3.0, 5: 3.0, 7: 20.0},
        max_iter=1000
    ).fit(xtrain_tfidf, train_y)
    
    print("\nEvaluating SVM model...")
    # Evaluate SVM
    svm_predictions = model_svm.predict(xvalid_tfidf)
    accuracy = metrics.accuracy_score(svm_predictions, valid_y_encoded)
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(metrics.classification_report(valid_y_encoded, svm_predictions, target_names=Data['product'].unique()))
    
    # Save the SVM model
    print("\nSaving SVM model...")
    with open('customer_classification_model_svm.pkl', 'wb') as file:
        pickle.dump(model_svm, file)
    
    print("\nSuccess! Both models have been trained and saved.")
    
    # Example prediction
    print("\nExample prediction:")
    example_text = ['This account popped up on my credit and it is not mines. I have filled out all the correct docs to show that i am victim of identity thief.']
    example_features = tfidf_vect.transform(example_text)
    prediction_lr = model_lr.predict(example_features)
    print(f"Example complaint predicted as: {Data['product'].unique()[prediction_lr[0]]}")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")