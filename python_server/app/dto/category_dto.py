from pydantic import BaseModel, Field
from typing import List

class CategoryUpdateReq(BaseModel):
    placeOfUse: str
    entryAmount: int
    memo: str
    category: str
    occurredAt: str

class TransActionBulkReq(BaseModel):
    transActions: List[CategoryUpdateReq]