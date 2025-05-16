from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from dotenv import load_dotenv
from app.services.gemini import Gemini
from app.schemas.request_schemas import GeneratePoemRequest, ChainRequest
from app.routers import reasoning_router
from src.models import VpecQwen3
from src import helper
from utils import data_helper, evaluator
import configs as config

load_dotenv()
app = FastAPI()
gemini = Gemini(api_key=os.getenv('GEMINI_API_KEY'))
vpec = VpecQwen3()
vpec._load_model()
checkpoint = helper.load_checkpoint(
  model_dir=vpec.model_name + '_0',
  model=vpec.model,
  optimizer=vpec.optimizer,
  is_the_best=True
)
vpec.model = checkpoint['model']
vpec.optimizer = checkpoint['optimizer']

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
  
@app.post("/edit-poem/step")
async def generate_step(chain: ChainRequest):
  try:
    if len(chain.steps) == 0: 
      error_poem = f"<sop> {chain.original_poem} <eop>"
    elif len(chain.steps) == 1:
      step = data_helper.parse_step(step_str=str(chain.steps[-1].step_content))
      error_poem = f"<sop> {chain.steps[-1].edited_poem} <eop> <reasoning_memory> Tóm tắt ngữ cảnh: {step['desc']} <eois>"
      edited_poem = chain.original_poem
    else:
      step = data_helper.parse_step(step_str=str(chain.steps[-1].step_content))
      error_poem = f"<sop> {chain.steps[-1].edited_poem} <eop> {chain.steps[-1].error_poem.split('<eop>')[1]} Sửa lỗi {step['error']}: Thay \"{step['replace']}\" bằng \"{step['action']}\" ở dòng {step['line']} tại từ thứ {step['index']} <eois>"
      error_poem = data_helper.filter_reasoning_memory(sample=error_poem, max_reasoning_memory=config.MAX_REASONING_MEMORY)
    acceptable = False
    while not acceptable:
      print("> Generating step")
      step_content = vpec.__generate__(input_text=error_poem, num_return_sequences=1)[0]
      scores = evaluator.get_step_structure_score(error_poem=error_poem, step_content=step_content)
      if scores['structure_score'] == 1 and scores['actionability_score'] == 1: acceptable = True
    step = data_helper.parse_step(step_str=step_content)
    print(step)
    edited_poem = data_helper.apply_edit_poem(chain.steps[-1].edited_poem, step['action'], step['replace'], int(step['line']), int(step['index'])) if len(chain.steps) > 0 else chain.original_poem
    return {'status': "OK", "error_poem": error_poem, "step_content": step_content, "edited_poem": edited_poem}
  except Exception as e:
    print(e)
    raise HTTPException(status_code=500, detail=str(e))

# app.include_router(reasoning_router.router, prefix='/api/v1')