# Consumer Complaint Classification

## Project Overview

This project implements machine learning models to automatically classify consumer complaints into different product categories based on the complaint narrative text. The classification helps financial institutions route complaints to the appropriate departments for timely resolution.

## Problem Statement

Financial institutions receive numerous customer complaints daily across various product lines. Manual classification of these complaints is time-consuming and prone to errors. This project aims to automate the classification process using natural language processing and machine learning techniques.

## Dataset

The project uses the Consumer Financial Protection Bureau's consumer complaint database containing customer complaints about financial products and services. The dataset includes:

- **consumer_complaint_narrative**: The text of the complaint submitted by the customer
- **product**: The financial product category (target variable for classification)

Product categories include:
- Credit card
- Mortgage
- Student loan
- Money transfers
- Debt collection
- Bank account or service
- Payday loan
- Consumer Loan
- Credit reporting
- Other financial service
- Prepaid card

## Models Implemented

### 1. Logistic Regression
- Multinomial classification with class weights to handle imbalanced data
- TF-IDF vectorization for text feature extraction
- Hyperparameter tuning using nested cross-validation

### 2. Support Vector Machine (SVM)
- Linear SVM implementation for multi-class classification
- Class weights to handle imbalanced categories
- Performance comparable to Logistic Regression

### 3. Hybrid Classification System
- Combines rule-based classification with machine learning models
- Uses fuzzy matching and context analysis
- Weighted voting between different classification approaches
- Provides explanations for classification decisions

## Project Structure

- **ConsumerComplaintClassification.ipynb**: Jupyter notebook containing EDA and model development
- **run_classification.py**: Script to train and evaluate both models
- **hybrid_classifier.py**: Advanced implementation combining rule-based and ML approaches
- **demo_classifier.py**: Simple demo script to showcase the trained models
- **test_model.py**: Script for testing the models with custom input
- **consumer_complaints.csv**: Dataset containing consumer complaints
- **customer_classification_model_lr.pkl**: Trained Logistic Regression model
- **customer_classification_model_svm.pkl**: Trained SVM model

## Key Features

1. **Text Preprocessing**:
   - Case normalization
   - Punctuation removal
   - Stopword filtering
   - TF-IDF vectorization

2. **Model Evaluation**:
   - Accuracy, precision, recall, and F1-score metrics
   - Confusion matrices
   - ROC curves and AUC values
   - Cross-validation for robust performance estimation

3. **Class Imbalance Handling**:
   - Weighted classes in model training
   - Optimized for better performance on minority classes

4. **Rule-Based Enhancement**:
   - Keyword matching for obvious cases
   - Context analysis around keywords
   - Fuzzy matching for handling variations in terminology

## Model Performance

Both models achieve approximately 85% accuracy with the Logistic Regression model performing slightly better on minority classes due to optimized class weights.

### Logistic Regression:
- **Accuracy**: ~85%
- **Macro-avg Precision**: ~84%
- **Macro-avg Recall**: ~83%

### SVM:
- **Accuracy**: ~84%
- **Macro-avg Precision**: ~83%
- **Macro-avg Recall**: ~82%

### Hybrid System:
- **Accuracy**: ~87%
- **Improved performance on minority classes**
- **Better explainability of classification decisions**

## How to Use

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, textblob, flask

### Setup
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### Training Models
To train the models from scratch:
```bash
python run_classification.py
```

### Web Interface
The project includes a simple web-based interface for easy interaction with the hybrid classifier:

1. **Start the Flask web server**:
```bash
python app.py
```

2. **Access the web interface**:
Open your browser and go to `http://localhost:5000`

3. **Use the interface**:
   - Enter a complaint text in the form or click on one of the example complaints
   - Submit the form to get classification results
   - View detailed analysis including predictions from all models, confidence scores, and explanations

### Command Line Testing
You can also use the hybrid classifier directly from the command line:
```bash
python hybrid_classifier.py
```

### Quick Start with Pre-trained Models
For users who just want to use the pre-trained models without retraining:

1. **Clone the repository**:
```bash
git clone [repository_url]
cd Consumer-Complaint-Classification
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn nltk textblob
```

3. **Use the demo script**:
```bash
python demo_classifier.py
```

This will load the pre-trained models and make predictions on example complaints.

4. **Test with your own complaints**:

You can modify the `demo_classifier.py` file to include your own complaint text:
```python
# Add your own complaint texts
my_complaints = [
    "I have been paying my mortgage for 5 years and suddenly they claim I missed a payment",
    "My credit card company charged me fees that were not disclosed"
]

# The script will run predictions on these examples
```

Alternatively, you can use this code snippet for direct integration:

### Using Pre-trained Models
```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
with open('customer_classification_model_lr.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare vectorizer (must match training configuration)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(training_complaints)  # Use the same complaints used for training

# Make prediction
features = tfidf_vect.transform([complaint_text])
prediction = model.predict(features)[0]
```

## Docker Deployment

The project includes Docker configuration for easy deployment and isolation:

### Prerequisites
- Docker and Docker Compose installed on your system

### Option 1: Using Docker Compose (Recommended)

1. **Build and start the container**:
```bash
docker-compose up -d
```

2. **Access the web application**:
Open your browser and go to `http://localhost:5000`

3. **Stop the container**:
```bash
docker-compose down
```

### Option 2: Using Docker directly

1. **Build the Docker image**:
```bash
docker build -t complaint-classifier .
```

2. **Run the Docker container**:
```bash
docker run -p 5000:5000 complaint-classifier
```

3. **Access the web application**:
Open your browser and go to `http://localhost:5000`

## Potential Improvements

1. **Model Enhancement**:
   - Try transformer-based models like BERT for potentially higher accuracy
   - Experiment with ensemble methods beyond simple voting

2. **Feature Engineering**:
   - Incorporate sentiment analysis features
   - Add domain-specific features based on financial terminology

3. **Deployment**:
   - Create a RESTful API service for real-time classification
   - Develop a user interface for business analysts

## Conclusion

This project demonstrates the effectiveness of machine learning in automating the classification of consumer complaints. The hybrid approach combining rule-based and ML models shows the best performance, particularly for minority classes, while also providing explanations for its decisions.

## References

- Scikit-learn documentation: https://scikit-learn.org/
- Consumer Financial Protection Bureau: https://www.consumerfinance.gov/data-research/consumer-complaints/
