import sys
import torch
import math
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.tasks.config.utils import CONFIG

is_windows = sys.platform.startswith('win')

app = FastAPI(title="Local VLM OpenAI-Compatible API")

# Global model state
MODEL_CONTAINER = {}

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas (OpenAI Spec Compat)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list of content blocks (for vision)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 5

# ---------------------------------------------------------------------------
# 2. Windows HF Model Initializer & Inference Function
# ---------------------------------------------------------------------------
def load_hf_model(model_name: str):
    if "model" not in MODEL_CONTAINER:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        print(f"[Windows HF Server] Loading model {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        MODEL_CONTAINER["model"] = model
        MODEL_CONTAINER["processor"] = processor
        MODEL_CONTAINER["tokenizer"] = processor.tokenizer
    return MODEL_CONTAINER["model"], MODEL_CONTAINER["processor"], MODEL_CONTAINER["tokenizer"]


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if not is_windows:
        raise HTTPException(status_code=400, detail="Use native vLLM entrypoint on Linux.")

    model, processor, tokenizer = load_hf_model(request.model)

    # Format OpenAI messages for HuggingFace chat template
    formatted_messages = []
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    prompt_inputs = processor.apply_chat_template(
        formatted_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Perform inference with logprob extraction if requested (G-Eval requirement)
    with torch.inference_mode():
        outputs = model.generate(
            **prompt_inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            do_sample=True if request.temperature > 0 else False,
            return_dict_in_generate=True,
            output_scores=True  # Guarantees logprobs access for G-Eval
        )

    # Extract Generated Tokens
    input_length = prompt_inputs["input_ids"].shape[-1]
    gen_tokens = outputs.sequences[0][input_length:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # Format logprobs if requested by G-Eval
    logprobs_payload = None
    if request.logprobs and hasattr(outputs, "scores"):
        # Get logits for first generated token
        first_token_scores = outputs.scores[0][0]  # shape: (vocab_size,)
        log_probs = torch.log_softmax(first_token_scores, dim=-1)

        # Extract top-k logprobs
        top_k_values, top_k_indices = torch.topk(log_probs, k=request.top_logprobs)
        
        top_logprobs_list = []
        for val, idx in zip(top_k_values, top_k_indices):
            token_str = tokenizer.decode([idx.item()])
            top_logprobs_list.append({
                "token": token_str,
                "logprob": val.item()
            })
            
        logprobs_payload = {
            "content": [{
                "token": tokenizer.decode([gen_tokens[0].item()]),
                "logprob": log_probs[gen_tokens[0].item()].item(),
                "top_logprobs": top_logprobs_list
            }]
        }

    # Construct OpenAI API Response
    return {
        "id": "chatcmpl-windows-hf",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "logprobs": logprobs_payload,
            "finish_reason": "stop"
        }]
    }

# ---------------------------------------------------------------------------
# 3. Main Launch Method
# ---------------------------------------------------------------------------
def run_server(host="0.0.0.0", port=8000, model_name=CONFIG["vlm"]["world_model"]["model_name"]):
    if is_windows:
        print("[Launcher] Windows detected. Launching FastAPI/HF local API server...")
        # Pre-load model
        load_hf_model(model_name)
        uvicorn.run(app, host=host, port=port)
    else:
        print("[Launcher] Linux detected. Launch vLLM using terminal command:")
        print(f"python -m vllm.entrypoints.openai.api_server --model {model_name} --port {port}")

if __name__ == "__main__":
    run_server(port=8000)
