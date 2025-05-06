import os
import pickle
import re
import string

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from robust_model_trainer import FFN  # your FFN class

app = Flask(__name__)
MODEL_DIR = "model_dir"
HIDDEN_DIM = 512  # must match your training

def preprocess(text: str) -> str:
    t = text.lower()
    t = re.sub(r'https?://\S+|\S*@\S*\s?', "", t)
    return t.translate(str.maketrans("", "", string.punctuation))

def load_artifacts():
    # load vectorizer & encoder
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # infer dims
    input_dim = vectorizer.max_features or len(vectorizer.vocabulary_)
    n_classes = len(label_encoder.classes_)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FFN(input_dim, HIDDEN_DIM, n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=device))
    model.eval()
    return model, vectorizer, label_encoder, device

MODEL, VEC, ENCODER, DEVICE = load_artifacts()

EXAMPLES = {
    "Credit card fee":       "I keep being charged an annual fee on my credit card that I didn’t agree to.",
    "Mortgage payment":      "My mortgage servicer applied my payment to escrow instead of principal.",
    "Debt collection calls": "Debt collectors are calling me at work even after I asked them to stop.",
    "Credit report error":   "My credit report shows a loan I never took out and it’s hurting my score.",
    "Student loan fee":      "My student loan servicer refuses to apply my payments correctly and added late fees.",
    "Overdraft fees":        "I was hit with multiple overdraft fees when my account was clearly in credit."
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text = ""
    if request.method == "POST":
        # get complaint text from form
        text = request.form.get("complaint_text", "").strip()
        if text:
            proc = preprocess(text)
            vec = VEC.transform([proc]).toarray().astype(np.float32)
            tensor = torch.tensor(vec, dtype=torch.float32).to(DEVICE)  # Updated to use torch.tensor instead of from_numpy
            with torch.no_grad():
                logits = MODEL(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]  # Updated to use torch.nn.functional
                idx = int(np.argmax(probs))
                prediction = ENCODER.inverse_transform([idx])[0]
                confidence = float(probs[idx])
    return render_template(
        "index.html",
        examples=EXAMPLES,
        complaint_text=text,
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
