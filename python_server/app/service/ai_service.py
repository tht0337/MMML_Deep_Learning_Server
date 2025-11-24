import os
import csv
import pandas as pd
from python_server.app.dto.category_dto import CategoryUpdateReq
from python_server.app.ml_training.user_finetune import (
    save_user_feedback,
    run_finetune,
    LOG_PATH,   # 기존 학습 데이터 경로
)
import torch
from python_server.app.config.category_config import CategoryConfig
from python_server.app.ml_training.train_gru import BiGRUTextClassifier

# =========================================
# 경로 설정 (항상 python_server/app 기준으로 고정)
# =========================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)

ML_TRAINING_DIR = os.path.join(APP_DIR, "ml_training")
USER_DIR = os.path.join(ML_TRAINING_DIR, "user_data")
# user_data = app/ml_training/user_data
CSV_PATH = os.path.join(USER_DIR, "correction_log.csv")
print("📁 USER_DIR:", USER_DIR)
print("📁 CSV_PATH:", CSV_PATH)
# user_data 폴더 자동 생성
os.makedirs(USER_DIR, exist_ok=True)


# =========================================
# 1. 사용자 피드백 저장
# =========================================
def save_user_feedback_service(req: CategoryUpdateReq):
    """
    유저가 수정한 카테고리 정보를 CSV에 append 저장
    """
    # correction_log.csv가 처음 생성되는 경우 헤더 추가
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # 첫 생성 시 헤더 작성
        if write_header:
            writer.writerow(["placeOfUse", "entryAmount", "memo", "category", "occurredAt"])

        # 데이터 추가
        writer.writerow([
            req.placeOfUse,
            req.entryAmount,
            req.memo,
            req.category,
            req.occurredAt
        ])

    return {"status": "saved", "data": req.model_dump()}


# =========================================
# 2. Fine-tune 준비 여부 확인
# =========================================
def check_finetune_ready_service():
    """
    유저 CSV 또는 기존 로그 CSV에서 데이터 개수 확인
    """
    if not os.path.exists(CSV_PATH):
        return {"count": 0, "ready": False}

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    count = len(df)

    return {
        "count": count,
        "ready": count >= 20  # 조건: 20건 이상이면 될 수 있음
    }


# =========================================
# 3. Fine-tune 실행
# =========================================
def run_finetune_service():
    """
    실제 fine-tune 실행
    """
    try:
        run_finetune()  # 외부 함수 호출
        return {"status": "ok", "msg": "fine-tune completed"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

MAX_LEN = 20

def encode_chars(text, char_to_idx):
    if not text or str(text).lower() == "nan":
        text = "<EMPTY>"

    ids = [char_to_idx.get(ch, 0) for ch in text[:MAX_LEN]]
    while len(ids) < MAX_LEN:
        ids.append(0)

    return torch.tensor([ids], dtype=torch.long)


def predict(price, merchant, memo=""):

    # 1️⃣ 룰 기반 먼저 처리
    rule = CategoryConfig.rule_based(merchant, memo)
    if rule is not None:
        return rule

    # 2️⃣ 모델 파일 불러오기
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))  # service/ 에서 한 단계 위로
    )
    MODEL_DIR = os.path.join(BASE_DIR, "ml_training", "model")

    MODEL_PATH = os.path.join(MODEL_DIR, "char_gru_classifier.pth")
    ENC_PATH = os.path.join(MODEL_DIR, "char_gru_encoders.pth")

    enc = torch.load(ENC_PATH, map_location="cpu", weights_only=False)
    char_to_idx = enc["char_to_idx"]
    category_encoder = enc["category_encoder"]

    price_tensor = torch.tensor([[price / 100000]], dtype=torch.float32)
    merchant_tensor = encode_chars(merchant, char_to_idx)
    memo_tensor = encode_chars(memo, char_to_idx)

    vocab_size = len(char_to_idx)
    num_classes = len(category_encoder.classes_)

    model = BiGRUTextClassifier(vocab_size, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(price_tensor, merchant_tensor, memo_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return category_encoder.inverse_transform([pred_idx])[0]