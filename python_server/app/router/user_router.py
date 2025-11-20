import os
from fastapi import APIRouter
import pandas as pd

from python_server.app.dto.category_dto import CategoryUpdateReq
from python_server.app.service.ai_service import (
    save_user_feedback_service,
    check_finetune_ready_service,
    run_finetune_service
)

router = APIRouter(prefix="/ai")


# 1. 사용자 카테고리 업데이트
@router.post("/update-category")
async def update_category(req: CategoryUpdateReq):
    return save_user_feedback_service(req)


# 2. 사용자 데이터 개수 확인
@router.get("/finetune-ready")
def finetune_ready():
    return check_finetune_ready_service()


# 3. 실제 Fine-tune 수행
@router.post("/finetune")
def finetune():
    return run_finetune_service()
