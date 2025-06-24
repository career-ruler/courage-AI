from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

lora_path = "/Users/dgsw8th20/Desktop/COURAGE/AI_SERVER/koalpaca-feedback-lora/checkpoint-363"

# config = PeftConfig.from_pretrained(lora_path)

# # GPU 있으면 4bit + fp16 + device_map=auto 로 로드, 없으면 CPU float32 모드로
# if torch.cuda.is_available():
#     base_model = AutoModelForCausalLM.from_pretrained(
#         config.base_model_name_or_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         load_in_4bit=True
#     )
#     model = PeftModel.from_pretrained(base_model, lora_path).to("cuda").eval()
# else:
#     base_model = AutoModelForCausalLM.from_pretrained(
#         config.base_model_name_or_path,
#         torch_dtype=torch.float32,
#     )
#     model = PeftModel.from_pretrained(base_model, lora_path).eval()

# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)

# class PromptRequest(BaseModel):
#     prompt: str

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 시에는 제한하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


class PromptQusestRequest(BaseModel):
    category: str
    question: str

# @app.post("/generate")
# async def generate_text(request: PromptRequest):
#     inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=512)
#     inputs.pop("token_type_ids", None)

#     if torch.cuda.is_available():
#         inputs = {k: v.to("cuda") for k, v in inputs.items()}

#     logger.info("Start generating...")
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     logger.info("Generation completed!")

#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"response": result}

@app.post("/generate_question")
async def generate_question(request: PromptQusestRequest):
    # 사용자에게 카테고리 받기
    category = request.category
    # 카테고리 받은걸로 질문 dataframe에서 질문 가져오기
    if category == "Ai":
        df = pd.read_csv("/Users/dgsw8th20/Desktop/COURAGE/AI_SERVER/AI_dataset.csv")
        # df에서 랜덤으로 질문 하나 가져오기
        question = df.sample(1).iloc[0]['질문']
    # 질문 사용자에게 반환
    return {"question": question}

class PromptAnswerRequest(BaseModel):
    answer: str

@app.post("/generate_answer")
async def generate_answer(request: PromptAnswerRequest):
    # 사용자에게 답변 받기
    answer = "추가적으로 딥러닝이 머신러닝보다 더 다양한 분야에서 사용되는 이유를 언급해 주세요."
    
    return {"answer": answer}




