from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Server Running!"}

# 🛠 Render 헬스체크용 HEAD 추가
@app.head("/")
def head():
    return 200