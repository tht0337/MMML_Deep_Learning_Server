import os
import glob
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from python_server.app.config.category_config import CategoryConfig

#카테고리
category_encoder = LabelEncoder()
category_encoder.fit(CategoryConfig.CATEGORIES)

# ================================
# Dataset
# ================================
class TransactionDataset(Dataset):
    def __init__(self, df, category_encoder, char_to_idx, max_len=20):
        self.price = df["price"].astype("float32").values
        self.merchant = df["merchant"].astype(str).values
        self.memo = df["memo"].astype(str).values
        normalized = df["category"].apply(CategoryConfig.normalize)
        self.labels = category_encoder.transform(normalized)

        self.char_to_idx = char_to_idx
        self.max_len = max_len

    def encode_chars(self, text):
        if not text or str(text).lower() == "nan":
            text = "<EMPTY>"

        ids = [self.char_to_idx.get(ch, 0) for ch in text[:self.max_len]]

        while len(ids) < self.max_len:
            ids.append(0)

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        price_tensor = torch.tensor([self.price[idx] / 100000], dtype=torch.float32)
        merchant_tensor = self.encode_chars(self.merchant[idx])
        memo_tensor = self.encode_chars(self.memo[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return price_tensor, merchant_tensor, memo_tensor, label_tensor



# ================================
# BiGRU MODEL
# ================================
class BiGRUTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=32, hidden_dim=64):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # merchant GRU (hidden*2) + memo GRU (hidden*2) + price(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        emb = self.emb(x)                  # (B, L, emb)
        _, h = self.gru(emb)               # h: (2, B, hidden)
        h_forward = h[-2]                  # (B, hidden)
        h_backward = h[-1]                 # (B, hidden)
        return torch.cat([h_forward, h_backward], dim=1)

    def forward(self, price, merchant_chars, memo_chars):
        merchant_vec = self.encode(merchant_chars)
        memo_vec = self.encode(memo_chars)

        x = torch.cat([price, merchant_vec, memo_vec], dim=1)
        return self.fc(x)



# ================================
# TRAIN FUNCTION
# ================================
def train_all_csv():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    MODEL_PATH = os.path.join(BASE_DIR, "model", "char_gru_classifier.pth")
    ENC_PATH = os.path.join(BASE_DIR, "model", "char_gru_encoders.pth")

    print("📌 DATA_DIR =", DATA_DIR)

    # -------------------------
    # 1) Load all CSV files
    # -------------------------
    csv_files = glob.glob(os.path.join(DATA_DIR, "category_*_50k.csv"))
    print("🔍 Found CSV:", csv_files)

    dfs = []
    for path in csv_files:
        print("➡ Loading", path)
        dfs.append(pd.read_csv(path, encoding="utf-8-sig"))

    df = pd.concat(dfs, ignore_index=True)
    print("📌 Total Rows Loaded:", len(df))  # 약 550,000 rows

    # -------------------------
    # 2) Build Encoders
    # -------------------------

    # Build character dictionary
    chars = set()
    for text in list(df["merchant"]) + list(df["memo"]):
        for ch in str(text):
            chars.add(ch)

    char_list = sorted(list(chars))

    char_to_idx = {ch: i+1 for i, ch in enumerate(char_list)}
    char_to_idx["<EMPTY>"] = 0

    vocab_size = len(char_to_idx)
    num_classes = len(category_encoder.classes_)

    print("📌 Vocab Size:", vocab_size)
    print("📌 Num Classes:", num_classes)

    # -------------------------
    # 3) Split
    # -------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = TransactionDataset(train_df, category_encoder, char_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # -------------------------
    # 4) Model
    # -------------------------
    model = BiGRUTextClassifier(vocab_size, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("🔥 Training Started")

    # -------------------------
    # 5) Train Loop
    # -------------------------
    EPOCHS = 20
    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0

        for price, merchant, memo, label in train_loader:
            opt.zero_grad()
            pred = model(price, merchant, memo)
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # -------------------------
    # 6) Save Model + Encoders
    # -------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save({
        "category_encoder": category_encoder,
        "char_to_idx": char_to_idx,
    }, ENC_PATH)

    print("🎉 Model Saved:", MODEL_PATH)
    print("🎉 Encoders Saved:", ENC_PATH)



if __name__ == "__main__":
    train_all_csv()
