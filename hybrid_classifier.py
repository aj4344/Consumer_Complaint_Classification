import pickle
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

# Change the NLTK imports and download process
import nltk
try:
    # Try to download punkt if not already downloaded
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
except Exception as e:
    # Fallback if NLTK download fails
    print(f"Warning: NLTK download failed: {str(e)}")
    print("Using simple string tokenizer instead of NLTK tokenizer")
    
    # Define simple fallback tokenization functions
    def word_tokenize(text):
        return text.split()
        
    def ngrams(tokens, n):
        return [tokens[i:i+n] for i in range(len(tokens) - n + 1)]

def preprocess_text(text):
    """Apply text preprocessing to standardize input text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def similar(a, b):
    """Calculate string similarity ratio using SequenceMatcher"""
    return SequenceMatcher(None, a, b).ratio()

def fuzzy_match(text, keyword, threshold=0.85):
    """Check if keyword is present in text using fuzzy matching"""
    if keyword in text:
        return True, 1.0
        
    words = text.split()
    keyword_parts = keyword.split()
    if len(keyword_parts) > 1:
        parts_present = 0
        for part in keyword_parts:
            if len(part) > 3:
                if any(part in word for word in words):
                    parts_present += 1
        
        if parts_present < len([p for p in keyword_parts if len(p) > 3]) / 2:
            return False, 0.0
            
        word_count = len(words)
        for i in range(word_count):
            if i + len(keyword_parts) > word_count:
                break
                
            phrase = " ".join(words[i:i+len(keyword_parts)])
            similarity = similar(phrase, keyword)
            if similarity >= threshold:
                return True, similarity
                
        return False, 0.0
    else:
        best_match = 0.0
        for word in words:
            if len(word) > 2 and len(keyword) > 2:
                similarity = similar(word, keyword)
                if similarity >= threshold:
                    return True, similarity
                best_match = max(best_match, similarity)
                
        return False, best_match

def find_context_phrases(text, keyword, window_size=5):
    """Find context phrases around keywords"""
    if keyword not in text:
        return []
        
    words = text.split()
    keyword_parts = keyword.split()
    contexts = []
    
    for i in range(len(words) - len(keyword_parts) + 1):
        if " ".join(words[i:i+len(keyword_parts)]) == keyword:
            start = max(0, i - window_size)
            end = min(len(words), i + len(keyword_parts) + window_size)
            context = " ".join(words[start:end])
            contexts.append(context)
            
    return contexts

def analyze_context(contexts, positive_indicators, negative_indicators):
    """Analyze context phrases for positive or negative indicators"""
    if not contexts:
        return 0
        
    score = 0
    for context in contexts:
        for indicator in positive_indicators:
            if indicator in context:
                score += 1
                
        for indicator in negative_indicators:
            if indicator in context:
                score -= 1
                
    return score / len(contexts)

def custom_tokenize_text(text):
    """Safe tokenization function that handles errors"""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()

def custom_ngrams(tokens, n):
    """Safe ngrams function that handles errors"""
    try:
        return list(ngrams(tokens, n))
    except Exception:
        return [tokens[i:i+n] for i in range(len(tokens) - n + 1) if i+n <= len(tokens)]

def rule_based_classifier(text, categories):
    """Enhanced rule-based classifier using keywords, fuzzy matching, and context analysis"""
    text = text.lower()
    
    keywords = {
        "Credit card": ["credit card", "creditcard", "card fee", "annual fee", "card account", 
                      "card payment", "card charge", "card interest", "card statement"],
        "Mortgage": ["mortgage", "foreclosure", "home loan", "house payment", "mortgage payment",
                    "mortgage interest", "mortgage servicer", "mortgage company", "refinance"],
        "Student loan": ["student loan", "student debt", "education loan", "student loan servicer",
                       "student loan payment", "student loan interest", "student loan forgiveness"],
        "Money transfers": ["money transfer", "wire transfer", "money sent", "money wired",
                          "transfer service", "money transmitter", "remittance"],
        "Debt collection": ["debt collector", "debt collection", "collecting debt", "collection agency",
                          "collection call", "collection notice", "debt collection practice"],
        "Bank account or service": ["bank account", "checking account", "savings account", "bank fee", 
                                  "bank service", "atm", "debit card", "overdraft"],
        "Payday loan": ["payday loan", "payday advance", "cash advance", "short term loan",
                      "payday lender", "high interest loan", "predatory loan"],
        "Consumer Loan": ["personal loan", "consumer loan", "auto loan", "car loan", "vehicle loan",
                       "installment loan", "loan application"],
        "Credit reporting": ["credit report", "credit score", "credit bureau", "credit history", 
                          "credit monitoring", "credit file", "credit dispute", "credit reporting agency"],
        "Other financial service": ["financial advisor", "financial planning", "investment", 
                               "financial product", "financial service"],
        "Prepaid card": ["prepaid card", "stored value card", "gift card", "prepaid account", 
                        "reloadable card"]
    }
    
    context_indicators = {
        "Credit card": {
            "positive": ["fee", "charge", "payment", "statement", "billing", "interest", "annual", 
                        "credit limit", "transaction", "purchase", "authorized", "unauthorized"],
            "negative": ["mortgage", "student", "payday", "personal"]
        },
        "Mortgage": {
            "positive": ["home", "house", "property", "foreclosure", "closing", "escrow", "refinance",
                        "loan officer", "interest rate", "payment", "principal", "lender", "loan"],
            "negative": ["credit card", "student loan", "payday", "auto loan"]
        },
        "Student loan": {
            "positive": ["student", "education", "school", "college", "university", "federal", "servicer", 
                        "deferment", "forbearance", "graduate", "degree", "forgiveness", "payment"],
            "negative": ["mortgage", "credit card", "auto loan", "payday"]
        },
        "Money transfers": {
            "positive": ["send", "sent", "received", "transfer", "wire", "remittance", "recipient",
                        "international", "western union", "moneygram", "paypal", "zelle"],
            "negative": ["mortgage", "credit card", "student loan"]
        },
        "Debt collection": {
            "positive": ["collector", "calling", "harassment", "harassing", "threaten", "threatening",
                        "collect", "validation", "letter", "cease", "desist", "attorney", "lawsuit"],
            "negative": ["mortgage payment", "loan application", "loan approval"]
        },
        "Bank account or service": {
            "positive": ["account", "bank", "deposit", "withdraw", "transaction", "branch", "online banking",
                        "mobile banking", "overdraft", "fee", "checking", "savings"],
            "negative": ["mortgage loan", "student loan", "credit card"]
        },
        "Payday loan": {
            "positive": ["payday", "short term", "quick cash", "advance", "high interest", "rollover",
                        "extension", "fast", "emergency", "predatory"],
            "negative": ["mortgage", "student loan", "credit card"]
        },
        "Consumer Loan": {
            "positive": ["personal", "auto", "car", "vehicle", "finance", "installment", "approval",
                        "application", "denied", "interest rate", "term"],
            "negative": ["mortgage", "student loan", "credit card", "payday"]
        },
        "Credit reporting": {
            "positive": ["report", "score", "bureau", "equifax", "experian", "transunion", "monitoring",
                        "dispute", "error", "incorrect", "information", "file", "inquiry"],
            "negative": ["mortgage loan", "student loan", "credit card"]
        },
        "Other financial service": {
            "positive": ["advisor", "investment", "portfolio", "planning", "strategy", "retirement",
                        "fund", "stock", "bond", "401k", "ira", "wealth", "management"],
            "negative": ["mortgage", "student loan", "credit card"]
        },
        "Prepaid card": {
            "positive": ["prepaid", "load", "reload", "balance", "activate", "activation", "netspend",
                        "green dot", "bluebird", "stored value", "gift"],
            "negative": ["credit card", "mortgage", "student loan"]
        }
    }
    
    scores = {category: 0 for category in categories}
    matched_keywords = {category: [] for category in categories}
    
    for category, words in keywords.items():
        if category in categories:
            for word in words:
                is_match, similarity = fuzzy_match(text, word, threshold=0.85)
                if is_match:
                    keyword_score = len(word.split()) * similarity * 2
                    
                    contexts = find_context_phrases(text, word) if word in text else []
                    
                    if contexts and category in context_indicators:
                        context_score = analyze_context(
                            contexts, 
                            context_indicators[category]["positive"],
                            context_indicators[category]["negative"]
                        )
                        keyword_score *= (1 + 0.5 * context_score)
                        
                    scores[category] += keyword_score
                    matched_keywords[category].append((word, similarity, keyword_score))
    
    tokens = custom_tokenize_text(text)
    two_grams = custom_ngrams(tokens, 2)
    three_grams = custom_ngrams(tokens, 3)
    
    two_gram_phrases = [' '.join(gram) for gram in two_grams]
    three_gram_phrases = [' '.join(gram) for gram in three_grams]
    
    phrase_indicators = {
        "Credit card": ["credit card", "card fee", "card payment", "card interest", "card statement"],
        "Mortgage": ["mortgage loan", "mortgage payment", "mortgage interest", "mortgage company"],
        "Student loan": ["student loan", "loan servicer", "loan payment", "loan interest"],
        "Money transfers": ["money transfer", "wire transfer", "send money", "receive money"],
        "Debt collection": ["debt collector", "debt collection", "collection agency", "collection call"],
        "Bank account or service": ["bank account", "checking account", "savings account", "bank fee"],
        "Payday loan": ["payday loan", "payday advance", "short term", "high interest"],
        "Consumer Loan": ["personal loan", "auto loan", "car loan", "loan application"],
        "Credit reporting": ["credit report", "credit score", "credit bureau", "credit history"],
        "Other financial service": ["financial advisor", "financial planning", "investment advice"],
        "Prepaid card": ["prepaid card", "card balance", "reload card"]
    }
    
    for category, phrases in phrase_indicators.items():
        if category in categories:
            for phrase in phrases:
                if phrase in two_gram_phrases or phrase in three_gram_phrases:
                    scores[category] += 3
                    matched_keywords[category].append((phrase, 1.0, 3))
    
    top_categories = sorted([(category, score) for category, score in scores.items() if score > 0], 
                           key=lambda x: x[1], reverse=True)[:3]
    
    if top_categories:
        best_category, score = top_categories[0]
        return {
            'category': best_category,
            'score': score,
            'confidence': min(score / 10, 1.0),
            'top_categories': top_categories,
            'matched_keywords': matched_keywords.get(best_category, [])
        }
    else:
        return None

def load_models_and_data():
    """Load models, vectorizer and categories"""
    try:
        with open('customer_classification_model_lr.pkl', 'rb') as file:
            lr_model = pickle.load(file)
        with open('customer_classification_model_svm.pkl', 'rb') as file:
            svm_model = pickle.load(file)
        
        data = pd.read_csv('consumer_complaints.csv', encoding='latin-1', low_memory=False)
        data = data[['product', 'consumer_complaint_narrative']]
        data = data[pd.notnull(data['consumer_complaint_narrative'])]
        
        categories = data['product'].unique()
        
        tfidf_vect = TfidfVectorizer(
            analyzer='word', 
            token_pattern=r'\w{1,}',
            max_features=5000, 
            stop_words='english'
        )
        tfidf_vect.fit(data['consumer_complaint_narrative'].apply(preprocess_text))
        
        return lr_model, svm_model, tfidf_vect, categories, data
    except Exception as e:
        print(f"Error loading models and data: {str(e)}")
        return None, None, None, None, None

def get_model_confidence(probabilities, prediction_idx):
    """Calculate confidence based on model probabilities"""
    if hasattr(probabilities, 'shape') and len(probabilities.shape) > 1:
        sorted_probs = np.sort(probabilities[0])[::-1]
        if len(sorted_probs) >= 2:
            margin = sorted_probs[0] - sorted_probs[1]
            return min(0.5 + margin * 2, 1.0)
    
    return 0.5

def hybrid_predict(complaint_text, lr_model, svm_model, tfidf_vect, categories):
    """Make prediction using enhanced hybrid approach with weighted voting"""
    # Apply rule-based classification with expanded features
    rule_result = rule_based_classifier(complaint_text, categories)
    
    # Preprocess and vectorize for ML models
    processed_text = preprocess_text(complaint_text)
    features = tfidf_vect.transform([processed_text])
    
    # Get ML model predictions
    lr_prediction_idx = lr_model.predict(features)[0]
    lr_prediction = categories[lr_prediction_idx]
    
    svm_prediction_idx = svm_model.predict(features)[0]
    svm_prediction = categories[svm_prediction_idx]
    
    # Get probabilities if available (for confidence calculation)
    try:
        lr_probabilities = lr_model.predict_proba(features)
        lr_confidence = get_model_confidence(lr_probabilities, lr_prediction_idx)
    except:
        # No probability method available, use fixed confidence
        lr_confidence = 0.5
    
    try:
        # SVM might not have predict_proba depending on implementation
        svm_probabilities = svm_model.decision_function(features)
        svm_confidence = get_model_confidence(svm_probabilities, svm_prediction_idx)
    except:
        # No probability method available, use fixed confidence
        svm_confidence = 0.5
    
    # Initialize weights for each method - IMPROVED WEIGHTS
    # Rule-based gets higher weight since it's more accurate for obvious cases
    weights = {
        'rule': 0.6,   # Increased from 0.5
        'lr': 0.2,     # Decreased from 0.25
        'svm': 0.2     # Decreased from 0.25
    }
    
    # Store all predictions with their weights and confidence
    predictions = {}
    
    # Add rule-based prediction if available
    if rule_result:
        rule_category = rule_result['category']
        rule_confidence = rule_result['confidence']
        
        # IMPROVEMENT: If confidence is very high and has strong keyword matches, boost weight
        if rule_confidence > 0.8 and len(rule_result['matched_keywords']) >= 2:
            boost_factor = 1.2
            weights['rule'] *= boost_factor
            weights['lr'] /= boost_factor
            weights['svm'] /= boost_factor
        
        predictions[rule_category] = predictions.get(rule_category, 0) + (weights['rule'] * rule_confidence)
    
    # Add ML model predictions
    predictions[lr_prediction] = predictions.get(lr_prediction, 0) + (weights['lr'] * lr_confidence)
    predictions[svm_prediction] = predictions.get(svm_prediction, 0) + (weights['svm'] * svm_confidence)
    
    # Get all predictions sorted by score
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top prediction
    final_prediction, score = sorted_predictions[0]
    
    # IMPROVEMENT: Improved tie breaking
    # If there's a virtual tie (scores very close), prefer rule-based if available
    if len(sorted_predictions) > 1:
        second_best = sorted_predictions[1]
        if abs(score - second_best[1]) < 0.05:  # If scores are within 5%
            # It's virtually a tie, check if rule_result exists and matches either
            if rule_result and (rule_result['category'] == final_prediction or 
                               rule_result['category'] == second_best[0]):
                final_prediction = rule_result['category']
                score = predictions[final_prediction]
    
    # IMPROVEMENT: Better calibrated confidence levels
    if score > 0.7:
        confidence = "High"
    elif score > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # IMPROVEMENT: Generate explanation text
    explanation = generate_explanation(
        complaint_text, final_prediction, 
        rule_result, lr_prediction, svm_prediction,
        score, sorted_predictions
    )
    
    # Store results
    results = {
        'rule_based': rule_result['category'] if rule_result else "No strong keyword match",
        'rule_confidence': rule_result['confidence'] if rule_result else 0,
        'logistic_regression': lr_prediction,
        'lr_confidence': lr_confidence,
        'svm': svm_prediction,
        'svm_confidence': svm_confidence,
        'weighted_scores': {k: round(v, 2) for k, v in predictions.items()},
        'matched_keywords': rule_result['matched_keywords'] if rule_result else [],
        'explanation': explanation  # New field for explanation
    }
    
    return final_prediction, confidence, results

def generate_explanation(text, prediction, rule_result, lr_prediction, svm_prediction, score, sorted_predictions):
    """Generate a natural language explanation for the classification decision"""
    
    # Start with a basic explanation
    explanation = f"This complaint was classified as '{prediction}' "
    
    # Check agreement between methods
    methods_agree = []
    if rule_result and rule_result['category'] == prediction:
        methods_agree.append("keyword analysis")
    if lr_prediction == prediction:
        methods_agree.append("logistic regression model")
    if svm_prediction == prediction:
        methods_agree.append("SVM model")
    
    # Case 1: All methods agree
    if len(methods_agree) == 3:
        explanation += f"with high confidence because all three methods (keyword analysis, logistic regression, and SVM) agreed on this classification."
    
    # Case 2: Two methods agree
    elif len(methods_agree) == 2:
        explanation += f"because {' and '.join(methods_agree)} both identified this category."
    
    # Case 3: Only one method picked this (likely rule-based with high confidence)
    elif len(methods_agree) == 1:
        explanation += f"primarily based on {methods_agree[0]}."
        
        # If it was rule-based and there were keywords, mention them
        if methods_agree[0] == "keyword analysis" and rule_result and rule_result['matched_keywords']:
            top_keywords = [kw[0] for kw in rule_result['matched_keywords'][:2]]
            explanation += f" Key terms like '{', '.join(top_keywords)}' strongly indicate this category."
    
    # Add info about close alternatives if applicable
    if len(sorted_predictions) > 1:
        second_best = sorted_predictions[1]
        if second_best[1] > 0.2:  # Only mention alternatives with reasonable scores
            explanation += f" '{second_best[0]}' was also considered as a possibility."
    
    return explanation

def main():
    print("\n===== Enhanced Hybrid Consumer Complaint Classification =====\n")
    
    print("Loading models and data...")
    lr_model, svm_model, tfidf_vect, categories, data = load_models_and_data()
    
    if lr_model is None or tfidf_vect is None:
        print("Failed to load required components. Exiting.")
        return
    
    print(f"✓ Models and data loaded successfully")
    print(f"✓ Found {len(categories)} product categories")

    examples = [
        "I have been having issues with my credit card account. I was charged an annual fee of $95 that was not disclosed when I signed up for the card. I have called customer service multiple times and they refuse to refund the fee.",
        
        "The mortgage company is threatening foreclosure on my home despite my payments being on time and properly documented. I've provided bank statements showing the payments were made but they claim there are missing payments.",
        
        "My student loan servicer is not correctly applying my payments to the principal balance as requested. Each month I submit extra payments specifically designated to reduce principal, but they are being applied to future interest instead.",
        
        "I sent a money transfer of $2000 to my family member last month, but the recipient never received it. When I contacted customer service, they claim the money was delivered but cannot provide proof.",
        
        "A debt collector keeps calling me daily regarding a debt that isn't mine. They're threatening legal action and damaging my credit score for someone else's debt.",
        
        "My bank account was closed without any notice or explanation. I had a significant amount of money in the account and they mailed me a check for the balance but provided no reason for the closure.",
        
        "I applied for a personal loan and was denied despite having a credit score of 780 and sufficient income documentation. When I asked for an explanation, they would not provide specific reasons for the denial.",
        
        "The payday loan company is charging an interest rate of over 400% APR, which exceeds what is legally allowed in my state. They did not clearly disclose the total cost of the loan and are now threatening aggressive collection actions."
    ]

    print("\nProduct Categories:")
    for i, category in enumerate(categories):
        print(f"{i+1}. {category}")
    
    print("\n===== Test with Example Complaints =====")
    print("Here are some example complaints you can use:")
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"{example[:150]}..." if len(example) > 150 else example)
    
    while True:
        print("\n===== Test the Enhanced Hybrid Classifier =====")
        print("Options:")
        print("1. Use an example complaint (enter number 1-8)")
        print("2. Enter your own complaint text")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1, 2, or 3): ")
        
        if choice == '3':
            break
        elif choice == '1':
            try:
                example_num = int(input("Enter example number (1-8): "))
                if example_num < 1 or example_num > 8:
                    print("Invalid example number. Please enter a number between 1 and 8.")
                    continue
                complaint_text = examples[example_num-1]
                print(f"\nUsing example: {complaint_text[:100]}...")
            except ValueError:
                print("Please enter a valid number.")
                continue
        elif choice == '2':
            complaint_text = input("\nEnter your complaint text: ")
            if not complaint_text.strip():
                print("Empty input. Please enter some text.")
                continue
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue
        
        try:
            final_prediction, confidence, results = hybrid_predict(
                complaint_text, lr_model, svm_model, tfidf_vect, categories
            )
            
            print("\n===== Prediction Results =====")
            print(f"Input: {complaint_text[:100]}..." if len(complaint_text) > 100 else f"Input: {complaint_text}")
            
            print("\n== Individual Model Results ==")
            print(f"Rule-based classifier: {results['rule_based']} (confidence: {results['rule_confidence']:.2f})")
            print(f"Logistic Regression: {results['logistic_regression']} (confidence: {results['lr_confidence']:.2f})")
            print(f"SVM: {results['svm']} (confidence: {results['svm_confidence']:.2f})")
            
            print("\n== Weighted Scores ==")
            for category, score in sorted(results['weighted_scores'].items(), key=lambda x: x[1], reverse=True):
                print(f"{category}: {score:.2f}")
            
            print("\n== Matched Keywords ==")
            if results['matched_keywords']:
                for keyword, similarity, score in results['matched_keywords'][:5]:  # Show top 5 matches
                    print(f"'{keyword}' (similarity: {similarity:.2f}, score: {score:.2f})")
            else:
                print("No strong keyword matches found")
            
            print("\n== Final Hybrid Prediction ==")
            print(f"Classification: {final_prediction}")
            print(f"Confidence: {confidence}")
            
            # IMPROVEMENT: Add explanation to the output
            print("\n== Explanation ==")
            print(results['explanation'])
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nThank you for using the Enhanced Hybrid Consumer Complaint Classification system!")

if __name__ == "__main__":
    main()