from fastapi import FastAPI
from python_server.app.router.user_router import router as user_router

app = FastAPI()
app.include_router(user_router, prefix="/ai")

@app.get("/")
def home():
    return {"message": "AI Server Running!"}

# 🛠 Render 헬스체크용 HEAD 추가
@app.head("/")
def head():
    return 200