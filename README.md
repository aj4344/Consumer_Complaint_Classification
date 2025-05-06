# Consumer Complaint Classification 2.0

## Overview
This project is an advanced consumer complaint classifier that categorizes financial service complaints into specific product categories. It helps financial institutions quickly route customer complaints to the appropriate departments, saving time and improving customer service efficiency.

The system uses a hybrid approach combining:
1. A PyTorch deep learning model
2. Natural language processing techniques
3. Advanced text preprocessing

## Problem Statement
Each week the Consumer Financial Protection Bureau receives thousands of consumer complaints about financial products and services. Manually categorizing these complaints is time-consuming and inefficient.

## Solution
This application automatically classifies complaints into specific product categories using machine learning. It's a multiclass classification system that analyzes the text of the complaint and assigns it to the appropriate category, enabling rapid routing to the correct department.

## Features
- **Robust Text Classification**: Handles misspellings, informal language, and domain-specific terminology
- **Web Interface**: Simple UI for submitting complaints and receiving classifications
- **Docker Support**: Easy deployment in containerized environments
- **Detailed Confidence Scores**: Shows classification confidence to aid in decision making
- **Example Complaints**: Pre-loaded examples to demonstrate functionality

## Model Architecture
The system uses a Feed-Forward Neural Network (FFN) implemented in PyTorch with:
- TF-IDF vectorization for text feature extraction
- Two fully connected layers with ReLU activation
- Dropout regularization for better generalization
- Cross-entropy loss optimization

## Dataset
The model is trained on real banking and finance consumer complaint data samples from the Consumer Financial Protection Bureau.

Data source: [Consumer Complaints Dataset](https://github.com/shubhamchouksey/Consumer-Complaint-Classification/blob/master/consumer_complaints.csv.zip)

## Product Categories
The classifier can identify complaints related to:
- Credit cards
- Mortgages
- Student loans
- Debt collection
- Bank accounts/services
- Payday loans
- Consumer loans
- Credit reporting
- Money transfers
- Other financial services
- Prepaid cards

## Project Structure
```
Consumer-Complaint-Classification/
├── app.py                    # Flask web application
├── robust_model_trainer.py   # PyTorch model definition and training
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── model_dir/                # Saved model artifacts
│   ├── model.pt              # Trained PyTorch model
│   ├── vectorizer.pkl        # TF-IDF vectorizer
│   └── label_encoder.pkl     # Label encoder for categories
└── templates/                # Web UI templates
    └── index.html            # Main interface
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Flask
- scikit-learn
- Docker (for containerized deployment)

### Option 1: Local Installation

1. Clone this repository:
```bash
git clone https://github.com/aj4344/Consumer_Complaint_Classification.git
cd Consumer-Complaint-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

### Option 2: Docker Deployment

1. Clone this repository:
```bash
git clone https://github.com/aj4344/Consumer_Complaint_Classification.git
cd Consumer-Complaint-Classification
```

2. Build the Docker image:
```bash
docker build -t consumercomplaintclassification2.0 .
```

3. Run the Docker container:
```bash
docker run -p 5000:5000 --name complaint_classifier consumercomplaintclassification2.0
```

4. Access the application at:
```
http://localhost:5000
```

## Docker Commands Reference

- **Build image**:
  ```bash
  docker build -t consumercomplaintclassification2.0 .
  ```

- **Run container**:
  ```bash
  docker run -p 5000:5000 --name complaint_classifier consumercomplaintclassification2.0
  ```

- **Run in background**:
  ```bash
  docker run -d -p 5000:5000 --name complaint_classifier consumercomplaintclassification2.0
  ```

- **Stop container**:
  ```bash
  docker stop complaint_classifier
  ```

- **Remove container**:
  ```bash
  docker rm complaint_classifier
  ```

- **View logs**:
  ```bash
  docker logs complaint_classifier
  ```

## Usage
1. Open the web interface
2. Enter a customer complaint in the text field
3. Click "Classify" to process
4. View the classification result and confidence score

## Model Training
To train your own model on new data:

1. Ensure you have labeled complaint data in CSV format with 'product' and 'consumer_complaint_narrative' columns
2. Run the training script:
```bash
python robust_model_trainer.py --train your_data.csv --epochs 10 --batch 128 --hidden 512 --save-dir model_dir
```

## Performance
The model achieves:
- Accuracy: ~85% on test data
- F1 Score: ~0.84 (weighted)
- Processing time: <100ms per complaint

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
MIT

## Acknowledgments
- Based on the original work by [Shubham Chouksey](https://github.com/shubhamchouksey/Consumer-Complaint-Classification)
- Enhanced with PyTorch deep learning techniques
- Optimized for Docker deployment
