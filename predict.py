# predict.py
import argparse
import pickle
import torch
import numpy as np
import re, string

from robust_model_trainer import FFN  # our model class

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|\S*@\S*\s?', "", text)
    return text.translate(str.maketrans("", "", string.punctuation))

def load_artifacts(model_dir: str, hidden_dim: int):
    # 1) vectorizer & encoder
    with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # 2) model weights
    # we infer input_dim and n_classes from the artifacts
    input_dim = vectorizer.max_features or vectorizer.vocabulary_.__len__()
    n_classes = len(label_encoder.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FFN(input_dim, hidden_dim, n_classes).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.eval()
    return model, vectorizer, label_encoder, device

def predict_text(model, vectorizer, label_encoder, device, text: str):
    proc = preprocess(text)
    vec = vectorizer.transform([proc]).toarray().astype(np.float32)
    tensor = torch.from_numpy(vec).to(device)
    with torch.no_grad():
        logits = model(tensor)
        idx = logits.argmax(1).item()
    return label_encoder.inverse_transform([idx])[0]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Directory with model.pt, vectorizer.pkl, etc.")
    p.add_argument("--hidden",    type=int, default=512, help="Hidden layer size used at training")
    p.add_argument("--text",      type=str, required=True, help="Complaint text to classify")
    args = p.parse_args()

    model, vec, encoder, device = load_artifacts(args.model_dir, args.hidden)
    pred = predict_text(model, vec, encoder, device, args.text)
    print(f"\n📝 Complaint: {args.text}\n➡️ Predicted category: {pred}")
