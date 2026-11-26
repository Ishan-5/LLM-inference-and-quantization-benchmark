import math
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
TEXT = ("The capital of France is Paris. It is a beautiful city with many attractions. "
        "People visit it every year for its museums and food. ") * 8

for use_triton in (False, True):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoGPTQForCausalLM.from_quantized(
        MODEL, device_map="auto", use_triton=use_triton, trust_remote_code=True
    )
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    ids = tok(TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    with torch.inference_mode():
        logits = model(input_ids=ids.unsqueeze(0).to(device)).logits[0, :-1, :].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[1:].to(device), reduction="mean")
    print(f"use_triton={use_triton} ppl={math.exp(float(nll)):.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
