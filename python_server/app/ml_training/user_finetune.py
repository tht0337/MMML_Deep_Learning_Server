import os
import csv
import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from python_server.app.config.category_config import CategoryConfig
from python_server.app.ml_training.train_gru import BiGRUTextClassifier, TransactionDataset

# =========================================
# 📌 PATH
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_PATH = os.path.join(USER_DATA_DIR, "correction_log.csv")

# =========================================
# 📌 Fine-Tune 상태 저장 (FastAPI에서 조회 가능)
# idle / running / success / fail
# =========================================
FINE_TUNE_STATUS = {
    "status": "idle",
    "message": None,
    "timestamp": None
}

# =========================================
# 📌 사용자 피드백 저장 (CSV append)
# =========================================
def save_user_feedback(merchant, price, memo, category):
    row = {
        "placeOfUse": merchant,
        "entryAmount": price,
        "memo": memo,
        "category": category,
        "timestamp": datetime.now().isoformat()
    }

    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📝 사용자 수정 저장: {row}")


# =========================================
# 📌 사용자 데이터 로드 + 기존 학습데이터 병합
# =========================================
def load_user_finetune_dataset(original_df):
    if not os.path.exists(LOG_PATH):
        print("⚠ 사용자 수정 데이터 없음 → 기본 데이터로 학습")
        return original_df

    user_df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    print(f"🔄 사용자 데이터 {len(user_df)}개 병합")

    merged = pd.concat([original_df, user_df], ignore_index=True)
    return merged


# =========================================
# 📌 Fine-Tune 실행 (백그라운드에서 돌릴 로직)
# =========================================
def run_finetune():
    print("\n🔥 [Fine-tune] 사용자 기반 재학습 시작")
    FINE_TUNE_STATUS["status"] = "running"
    FINE_TUNE_STATUS["timestamp"] = datetime.now().isoformat()
    FINE_TUNE_STATUS["message"] = "학습 중..."

    try:
        # -------------------------
        # 1) 기존 학습 데이터 로드
        # -------------------------
        original_path = os.path.join(BASE_DIR, "data", "combined_train.csv")
        print(original_path)

        if not os.path.exists(original_path):
            raise Exception("기존 학습 데이터 combined_train.csv 없음")

        df = pd.read_csv(original_path, encoding="utf-8-sig")
        print(f"📌 기존 학습 데이터: {len(df)}행")

        # -------------------------
        # 2) 사용자 데이터 병합
        # -------------------------
        df = load_user_finetune_dataset(df)
        print(f"📌 전체 학습 데이터: {len(df)}행")

        # -------------------------
        # 3) 칼럼명 통일
        # 기존 데이터가 merchant/price일 수 있으므로 rename
        # -------------------------
        rename_map = {
            "merchant": "placeOfUse",
            "price": "entryAmount"
        }
        df = df.rename(columns=rename_map)

        required_cols = ["placeOfUse", "entryAmount", "memo", "category"]
        for col in required_cols:
            if col not in df.columns:
                raise Exception(f"Fine-tune 불가: '{col}' 컬럼이 존재하지 않음")

        # -------------------------
        # 4) 라벨 인코더 생성
        # -------------------------
        category_encoder = LabelEncoder()
        category_encoder.fit(CategoryConfig.CATEGORIES)

        # -------------------------
        # 5) 문자 인덱스 생성
        # -------------------------
        chars = set()
        for t in df["placeOfUse"].astype(str).tolist() + df["memo"].astype(str).tolist():
            for ch in t:
                chars.add(ch)

        char_list = sorted(list(chars))
        char_to_idx = {ch: i+1 for i, ch in enumerate(char_list)}
        char_to_idx["<EMPTY>"] = 0

        vocab_size = len(char_to_idx)
        num_classes = len(category_encoder.classes_)

        # -------------------------
        # 6) Train/Test split
        # -------------------------
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_dataset = TransactionDataset(train_df, category_encoder, char_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        # -------------------------
        # 7) 모델 로드
        # -------------------------
        model_path = os.path.join(MODEL_DIR, "char_gru_classifier.pth")
        model = BiGRUTextClassifier(vocab_size, num_classes)

        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print("📥 기존 모델 불러옴")
        except:
            print("⚠ 기존 모델 구조 불일치 → 새 모델 학습")

        # -------------------------
        # 8) Fine-Tune 학습Loop
        # -------------------------
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = nn.CrossEntropyLoss()

        print("🔥 Fine-tune 학습 시작")

        for epoch in range(6):
            total_loss = 0
            for price, merchant, memo, label in train_loader:
                opt.zero_grad()
                pred = model(price, merchant, memo)
                loss = loss_fn(pred, label)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

        # -------------------------
        # 9) 모델 저장
        # -------------------------
        torch.save(model.state_dict(), model_path)
        torch.save({
            "category_encoder": category_encoder,
            "char_to_idx": char_to_idx,
        }, os.path.join(MODEL_DIR, "char_gru_encoders.pth"))

        print("🎉 Fine-tune 완료 & 모델 업데이트됨")

        # 성공 상태 저장
        FINE_TUNE_STATUS["status"] = "success"
        FINE_TUNE_STATUS["message"] = "학습 완료"

        return True

    except Exception as e:
        print("❌ Fine-Tune 실패:", str(e))

        FINE_TUNE_STATUS["status"] = "fail"
        FINE_TUNE_STATUS["message"] = str(e)

        return False
