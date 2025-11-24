import os
import csv
import pandas as pd
from python_server.app.dto.category_dto import CategoryUpdateReq
from python_server.app.ml_training.user_finetune import (
    save_user_feedback,
    run_finetune,
    LOG_PATH,   # 기존 학습 데이터 경로
)

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
