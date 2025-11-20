import os
import pandas as pd
import csv
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

from python_server.app.config.category_config import CategoryConfig
from python_server.app.ml_training.train_gru import BiGRUTextClassifier, TransactionDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_PATH = os.path.join(USER_DATA_DIR, "correction_log.csv")

# ================ 데이터셋 세이브 ================
def save_user_feedback(merchant: str, price: int, memo: str, category: str):
    """
    사용자 카테고리 수정 데이터를 correction_log.csv에 저장하는 함수
    - merchant : 가맹점명
    - price    : 가격(정수)
    - memo     : 메모/상세내용
    - category : 사용자가 직접 선택한 카테고리
    """

    row = {
        "merchant": merchant,
        "price": price,
        "memo": memo,
        "category": category,
        "timestamp": datetime.now().isoformat()
    }

    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # 첫 저장 시 header 생성
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"📝 저장됨 → {row}")
    return True
# ================ 데이터셋 로더 (기존 train_gru 코드 재사용) ================
def load_user_finetune_dataset(original_df):
    """기존 학습 데이터(original_df)에 correction_log.csv 병합"""

    if not os.path.exists(LOG_PATH):
        print("⚠ 사용자 수정 데이터 없음. 기본 모델 유지.")
        return original_df

    user_df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    print(f"🔄 사용자 데이터 {len(user_df)}개 병합")

    merged = pd.concat([original_df, user_df], ignore_index=True)
    return merged


# ================ 실제 미세학습 로직 ================
def run_finetune():
    print("\n🔥 [Fine-tune] 사용자 데이터 기반 재학습 시작")

    # 1) 기존 학습데이터 로드
    original_path = os.path.join(BASE_DIR, "data", "combined_train.csv")
    if not os.path.exists(original_path):
        raise Exception("❌ 기존 학습 데이터 파일이 없습니다: combined_train.csv")

    df = pd.read_csv(original_path, encoding="utf-8-sig")
    print(f"📌 기존 학습 데이터: {len(df)}행")

    # 2) 사용자 수정데이터 병합
    df = load_user_finetune_dataset(df)
    print(f"📌 병합된 전체 데이터: {len(df)}행")

    # 3) 라벨/문자 인코더 생성
    category_encoder = LabelEncoder()
    category_encoder.fit(CategoryConfig.CATEGORIES)

    # 문자 인덱싱
    chars = set()
    for t in df["merchant"].astype(str).tolist() + df["memo"].astype(str).tolist():
        for ch in t:
            chars.add(ch)

    char_list = sorted(list(chars))
    char_to_idx = {ch: i+1 for i, ch in enumerate(char_list)}
    char_to_idx["<EMPTY>"] = 0

    vocab_size = len(char_to_idx)
    num_classes = len(category_encoder.classes_)

    # 4) 데이터 split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = TransactionDataset(train_df, category_encoder, char_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 5) 모델 로드 후 재학습
    model_path = os.path.join(MODEL_DIR, "char_gru_classifier.pth")
    model = BiGRUTextClassifier(vocab_size, num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("📥 기존 모델 불러옴")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    print("🔥 Fine-tune 학습 시작")

    for epoch in range(6):  # 사용자 데이터는 적으니 적당한 Epoch
        total_loss = 0
        for price, merchant, memo, label in train_loader:
            opt.zero_grad()
            pred = model(price, merchant, memo)
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # 6) 모델 저장
    torch.save(model.state_dict(), model_path)
    torch.save({
        "category_encoder": category_encoder,
        "char_to_idx": char_to_idx,
    }, os.path.join(MODEL_DIR, "char_gru_encoders.pth"))

    print("🎉 Fine-tune 완료 & 모델 업데이트됨")

    return True
