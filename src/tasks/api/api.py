import base64
import binascii
from contextlib import contextmanager
import mimetypes
import os
import tempfile
import torch
import math
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx

def sanitize_float(val: float) -> float:
    """Replace non-finite float values with finite approximations for JSON compliance."""
    if math.isinf(val):
        return -1e9 if val < 0 else 1e9
    if math.isnan(val):
        return 0.0
    return val

from src.tasks.api.llama_cpp_backend import (
    get_llama_cpp_server_url,
    start_llama_cpp_server,
)
from src.tasks.config.utils import apply_vlm_profile, resolve_vlm_profile
from src.tasks.environment import load_world_model

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
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 1024
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 10


@contextmanager
def _transformers_messages(messages):
    """Translate OpenAI video blocks to the format expected by Transformers."""
    temporary_files = []
    formatted_messages = []
    try:
        for message in messages:
            content = message.content
            if isinstance(content, list):
                formatted_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "video_url":
                        video_url = block.get("video_url")
                        url = video_url.get("url") if isinstance(video_url, dict) else video_url
                        if not isinstance(url, str):
                            raise HTTPException(
                                status_code=400,
                                detail="video_url must contain a string URL.",
                            )

                        if url.startswith("data:"):
                            header, separator, encoded_video = url.partition(",")
                            metadata = header[5:].split(";")
                            media_type = metadata[0].lower()
                            if (
                                not separator
                                or not media_type.startswith("video/")
                                or "base64" not in {part.lower() for part in metadata[1:]}
                            ):
                                raise HTTPException(
                                    status_code=400,
                                    detail="video_url data must be a base64-encoded video.",
                                )
                            try:
                                video_bytes = base64.b64decode(encoded_video, validate=True)
                            except (binascii.Error, ValueError) as exc:
                                raise HTTPException(
                                    status_code=400,
                                    detail="video_url contains invalid base64 data.",
                                ) from exc

                            suffix = mimetypes.guess_extension(media_type) or ".video"
                            with tempfile.NamedTemporaryFile(
                                suffix=suffix, delete=False
                            ) as video_file:
                                temporary_files.append(video_file.name)
                                video_file.write(video_bytes)
                            url = video_file.name

                        formatted_content.append({"type": "video", "url": url})
                    else:
                        formatted_content.append(block)
                content = formatted_content

            formatted_messages.append({"role": message.role, "content": content})

        yield formatted_messages
    finally:
        for path in temporary_files:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# 2. Windows HF Model Initializer & Inference Function
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    profile = resolve_vlm_profile(request.model)
    if profile["backend"] == "llama.cpp":
        server_url = get_llama_cpp_server_url(profile["id"])
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{server_url}/v1/chat/completions",
                json=request.dict(exclude_none=True),
            )
        if response.is_error:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

    model, processor, tokenizer = load_world_model(request.model)

    with _transformers_messages(request.messages) as formatted_messages:
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
        k_val = request.top_logprobs if request.top_logprobs is not None else 10
        top_k_values, top_k_indices = torch.topk(log_probs, k=k_val)
        
        top_logprobs_list = []
        for val, idx in zip(top_k_values, top_k_indices):
            token_str = tokenizer.decode([idx.item()])
            top_logprobs_list.append({
                "token": token_str,
                "logprob": sanitize_float(val.item())
            })
            
        logprobs_payload = {
            "content": [{
                "token": tokenizer.decode([gen_tokens[0].item()]),
                "logprob": sanitize_float(log_probs[gen_tokens[0].item()].item()),
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
def run_server(host="0.0.0.0", port=5015, model_name=None):
    print("[Launcher] Starting local VLM API server...")
    profile = apply_vlm_profile() if model_name is None else resolve_vlm_profile(model_name)
    if profile["backend"] == "llama.cpp":
        start_llama_cpp_server(profile)
    else:
        load_world_model(profile["repo_id"])
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server(port=5015)
