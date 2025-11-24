import os
from fastapi import APIRouter, BackgroundTasks
import pandas as pd

from python_server.app.dto.category_dto import TransActionBulkReq
from python_server.app.ml_training.user_finetune import run_finetune, FINE_TUNE_STATUS
from python_server.app.service.ai_service import (
    save_user_feedback_service,
    check_finetune_ready_service,
    run_finetune_service,
    predict
)

router = APIRouter()

# 사용자 카테고리 업데이트(여기서는 유저의 변경 사항을 csv로 저장)
@router.post("/update-category")
async def update_category(req: TransActionBulkReq):

    saved = 0
    failed = 0
    failed_items = []

    for idx, item in enumerate(req.transActions):
        try:
            save_user_feedback_service(item)
            saved += 1

        except Exception as e:
            failed += 1
            failed_items.append({
                "index": idx,                     # 실패한 row 번호
                "reason": str(e),                 # 실제 에러 메시지
                "rawData": item.model_dump()      # 해당 row 전체 데이터
            })

    # 전체 status 구성
    response = {
        "status": "success" if failed == 0 else "partial",
        "count": saved,
        "saved": saved,
        "failed": failed
    }

    if failed > 0:
        response["failedItems"] = failed_items

    return response


# 사용자 데이터 개수 확인
@router.get("/finetune-ready")
def finetune_ready():
    return check_finetune_ready_service()


# Fine-tune 수행
@router.post("/finetune")
def finetune():
    return run_finetune_service()

# 유저 거래 내역으로 카테고리 분류
@router.post("/classify-transaction")
async def classify_transaction(req: TransActionBulkReq):

    results = []
    failed = 0
    failed_items = []

    for idx, item in enumerate(req.transActions):
        try:
            predicted = predict(
                price=item.entryAmount,
                merchant=item.placeOfUse,
                memo=item.memo
            )

            results.append({
                "placeOfUse": item.placeOfUse,
                "entryAmount": item.entryAmount,
                "memo": item.memo,
                "category": predicted,
                "occurredAt": item.occurredAt
            })

        except Exception as e:
            failed += 1
            failed_items.append({
                "index": idx,
                "reason": str(e),
                "rawData": item.model_dump()
            })

    response = {
        "status": "success" if failed == 0 else "partial",
        "count": len(results),
        "failed": failed,
        "results": results
    }

    if failed_items:
        response["failedItems"] = failed_items

    return response

@router.post("/fine-tune")
async def classify_transaction(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_finetune) # timeout 대비
    return {
        "status": "started",
        "message": "Fine-tune 작업이 백그라운드에서 실행됩니다."
    }

@router.get("/fine-tune/status")
def fine_tune_status():
    return FINE_TUNE_STATUS