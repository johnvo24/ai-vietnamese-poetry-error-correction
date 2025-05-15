from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from dotenv import load_dotenv
from app.services.gemini import Gemini
from app.schemas.request_schemas import GeneratePoemRequest
from app.routers import reasoning_router

load_dotenv()
app = FastAPI()
gemini = Gemini(api_key=os.getenv('GEMINI_API_KEY'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
  return JSONResponse(
    status_code=404,
    content={"error":"Không tồn tại đường dẫn này má ơi!"}
  )

# Route cơ bản: Trang chính
@app.get("/")
def read_root():
  return {"message": "Welcome to VpecAI backend!"}

# Route cơ bản: Trang chính
@app.get("/models/", response_model=List[str])
def list_models():
  models = [
    gemini.name,
  ]
  return JSONResponse(content=models, status_code=200)

# API POST: Nhận dữ liệu từ client
@app.post("/generate-poem/")
async def create_item(req: GeneratePoemRequest):
  if req.model == gemini.name:
    model = gemini
  else:
    raise HTTPException(status_code=400, detail="Model not supported")
  try:
    content = model.__generate__(f"Sau đây tôi sẽ đưa yêu cầu tạo bài thơ lục bát: {req.prompt}.\nChỉ trả về bài thơ lục bát (câu 6, câu 8) từ 2 đến 6 câu.")
    content = content.replace("\n\n", "\n")
    title = gemini.__generate__(f"{content}. Đặt tên cho bài thơ này (chỉ ghi tên)")
    return {"title": title, "prompt": req.prompt,"content": content}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
# @app.post("/edit-poem/")
# async def 
  
# app.include_router(reasoning_router.router, prefix='/api/v1')