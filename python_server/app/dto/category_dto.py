from pydantic import BaseModel, Field
from typing import List,Optional

class CategoryUpdateReq(BaseModel):
    placeOfUse: Optional[str] = None
    entryAmount: Optional[int] = None
    memo: Optional[str] = None
    category: Optional[str] = None
    occurredAt: Optional[str] = None

class TransActionBulkReq(BaseModel):
    transActions: List[CategoryUpdateReq]