# robust_model_trainer.py

import os
import time
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, X_sparse, y_array):
        self.X = X_sparse
        self.y = y_array

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].toarray().squeeze().astype(np.float32)
        y = np.int64(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class RobustComplaintClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=5,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.model = None

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+|\S*@\S*\s?', "", text)
        return text.translate(str.maketrans("", "", string.punctuation))

    def load_data(self, path):
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df = df[["product", "consumer_complaint_narrative"]].dropna()
        df["text"] = df["consumer_complaint_narrative"].map(self.preprocess)
        y = self.encoder.fit_transform(df["product"])
        return train_test_split(
            df["text"], y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    def train(self, data_path, epochs=5, batch_size=128, hidden_dim=256, save_dir="model_dir"):
        os.makedirs(save_dir, exist_ok=True)
        start = time.time()

        X_train_txt, X_test_txt, y_train, y_test = self.load_data(data_path)
        X_train = self.vectorizer.fit_transform(X_train_txt)
        X_test = self.vectorizer.transform(X_test_txt)
        input_dim = X_train.shape[1]
        n_classes = len(self.encoder.classes_)

        train_ds = TextDataset(X_train, y_train)
        test_ds = TextDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = FFN(input_dim, hidden_dim, n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(f"🔄 Training on {self.device} for {epochs} epochs…")
        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            avg_loss = total_loss / len(train_loader.dataset)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb = Xb.to(self.device)
                    logits = model(Xb)
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_labels.append(yb.numpy())
            preds = np.concatenate(all_preds)
            labs = np.concatenate(all_labels)
            acc = accuracy_score(labs, preds)
            f1 = f1_score(labs, preds, average="weighted")
            print(f"Epoch {ep}/{epochs} — loss: {avg_loss:.4f} — val_acc: {acc:.4f} — val_f1: {f1:.4f}")

        print("\n📊 Final evaluation:")
        print(classification_report(labs, preds, target_names=self.encoder.classes_, zero_division=0))

        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.encoder, f)

        print(f"\n✅ Training took {(time.time() - start)/60:.2f} min, artifacts in '{save_dir}/'")


# run_robust_model.py

import argparse
from robust_model_trainer import RobustComplaintClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train robust complaint classifier")
    parser.add_argument("--train", required=True, help="Path to consumer_complaints.csv")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--save-dir", default="model_dir", help="Directory to save model artifacts")
    args = parser.parse_args()

    clf = RobustComplaintClassifier()
    clf.train(
        data_path=args.train,
        epochs=args.epochs,
        batch_size=args.batch,
        hidden_dim=args.hidden,
        save_dir=args.save_dir
    )
