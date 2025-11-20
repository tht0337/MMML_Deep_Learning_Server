import os
import pandas as pd

from python_server.app.ml_training.user_finetune import (
    save_user_feedback,
    run_finetune,
    LOG_PATH
)


# 1. 사용자 피드백 저장 Service
def save_user_feedback_service(req):
    data = req.dict()

    save_user_feedback(
        merchant=data["merchant"],
        price=data["price"],
        memo=data["memo"],
        category=data["category"]
    )

    return {
        "status": "saved",
        "received": data
    }


# 2. Finetune 준비 여부 확인 Service
def check_finetune_ready_service():
    if not os.path.exists(LOG_PATH):
        return {"count": 0, "ready": False}

    df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    count = len(df)

    return {
        "count": count,
        "ready": count >= 20
    }


# 3. Fine-tune 실행 Service
def run_finetune_service():
    try:
        run_finetune()
        return {"status": "ok", "msg": "fine-tune completed"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}
