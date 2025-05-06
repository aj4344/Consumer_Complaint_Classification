from flask import Flask, render_template, request, jsonify
import sys
import os
from hybrid_classifier import load_models_and_data, hybrid_predict

app = Flask(__name__)

# Load the models and data
print("Loading models and data...")
lr_model, svm_model, tfidf_vect, categories, data = load_models_and_data()

if lr_model is None or tfidf_vect is None:
    print("Failed to load required components. Exiting.")
    sys.exit(1)

print("Models loaded successfully!")

@app.route('/')
def index():
    """Render the main page with the complaint form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the complaint text and return predictions"""
    try:
        # Get complaint text from the form
        complaint_text = request.form.get('complaint_text', '')
        
        if not complaint_text.strip():
            return jsonify({'error': 'Please enter a complaint text'})
        
        # Use the hybrid classifier to make prediction
        final_prediction, confidence, results = hybrid_predict(
            complaint_text, lr_model, svm_model, tfidf_vect, categories
        )
        
        # No need to convert confidence to float - keep it as a string
        return render_template(
            'index.html',  # Use the same template for results
            complaint_text=complaint_text,
            show_results=True,  # Flag to show the results section
            final_prediction=final_prediction,
            confidence=confidence,  # Keep as string
            rule_based=results['rule_based'],
            rule_confidence=f"{results['rule_confidence']:.2f}",
            logistic_regression=results['logistic_regression'],
            lr_confidence=f"{results['lr_confidence']:.2f}",
            svm=results['svm'],
            svm_confidence=f"{results['svm_confidence']:.2f}",
            weighted_scores=sorted(results['weighted_scores'].items(), key=lambda x: x[1], reverse=True),
            matched_keywords=results['matched_keywords'][:5] if results['matched_keywords'] else [],
            explanation=results['explanation']
        )
        
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    # Use 0.0.0.0 to allow external access (important for Docker)
    app.run(debug=False, host='0.0.0.0', port=5000)