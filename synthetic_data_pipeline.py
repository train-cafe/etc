import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
CACHE_DIR = "/group-volume/kang9.lee/models"
os.environ["HF_HOME"] = CACHE_DIR

import time
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [DataAugmentation] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing large-scale synthetic data augmentation pipeline...")
    logger.info("Configuring hardware devices and memory mapping...")

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"Created local cache directory for model weights: {CACHE_DIR}")

    model_id = "Qwen/Qwen2.5-72B-Instruct" 

    logger.info(f"Loading model checkpoint: {model_id} (bfloat16 precision)")
    logger.info(f"Using local cache directory: {CACHE_DIR}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=CACHE_DIR
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        logger.error(f"Failed to load model architecture: {e}")
        return

    logger.info("Model loaded successfully. Entering batch processing phase.")

    prompt_pool = [
        "Explain the detailed architecture of transformer models and write a comprehensive Python implementation from scratch.",
        "Write a 3000-word comprehensive essay about the socio-economic impacts of general artificial intelligence over the next 50 years.",
        "Generate a highly complex, advanced Python script that simulates a multi-agent reinforcement learning environment in detail."
    ]

    total_batches = 12000  
    start_time = time.time()

    for batch_idx in range(1, total_batches + 1):
        batch_start = time.time()
        
        prompt = prompt_pool[batch_idx % len(prompt_pool)]
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - batch_start
        
        if batch_idx % 10 == 0:
            est_remaining_hours = (latency * (total_batches - batch_idx)) / 3600
            logger.info(
                f"Processed batch {batch_idx}/{total_batches} | "
                f"Generated content len: {len(response)} chars | "
                f"Latency: {latency:.2f}s | "
                f"Est. remaining: {est_remaining_hours:.2f} hrs"
            )

    total_time = (time.time() - start_time) / 3600
    logger.info(f"Pipeline completed successfully. Total processing time: {total_time:.2f} hours.")

if __name__ == "__main__":
    main()
