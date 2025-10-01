from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from utils import model_with_tool
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request body model
class ChatRequest(BaseModel):
    # query: str
    chat_history: List[dict]  # [{"role": "human", "content": "..."}, {"role": "system", "content": "..."}]

# Response body model
class ChatResponse(BaseModel):
    answer: str
    # updated_history: List[dict]

@app.get("/")
def home():
    return {"message", "its working fine :)"}



@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):

    result = model_with_tool.rag_agent(request.chat_history)

    return ChatResponse(answer=result["content"])

