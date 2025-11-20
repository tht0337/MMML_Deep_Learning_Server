from pydantic import BaseModel

class CategoryUpdateReq(BaseModel):
    merchant: str
    price: int
    memo: str
    category: str