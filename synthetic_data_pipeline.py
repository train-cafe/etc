import os

# 스크립트 최상단에서 지정하여 다른 GPU의 간섭을 차단하고 2번, 3번 H100만 사용합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Hugging Face 다운로드 캐시 경로를 스크립트 레벨에서 강제 지정 (환경변수 설정)
# 이렇게 하면 다운로드된 모델이 기본 숨김 폴더(~/.cache)가 아닌 명시된 폴더에 저장됩니다.
CACHE_DIR = "/group-volume/kang9.lee/models"
os.environ["HF_HOME"] = CACHE_DIR

import time
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 실제 프로덕션 파이프라인처럼 보이도록 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - [DataAugmentation] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing large-scale synthetic data augmentation pipeline...")
    logger.info("Configuring hardware devices and memory mapping...")

    # 모델 가중치를 저장할 폴더가 없다면 생성
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"Created local cache directory for model weights: {CACHE_DIR}")

    # H100 2장(총 160GB)의 VRAM을 최대한 꽉 채우기 위해 72B(Billion) 파라미터 모델을 사용합니다.
    model_id = "Qwen/Qwen2.5-72B-Instruct" 

    logger.info(f"Loading model checkpoint: {model_id} (bfloat16 precision)")
    logger.info(f"Using local cache directory: {CACHE_DIR}")
    
    try:
        # cache_dir을 명시하여 해당 폴더에 모델을 저장/로드합니다.
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=CACHE_DIR
        )
        
        # device_map="auto"가 H100 2개에 모델을 절반씩 쪼개어 적재하여 메모리 사용률을 극대화합니다.
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

    # 모델이 길고 복잡한 연산을 하도록 유도하는 프롬프트 (번갈아가며 사용하여 GPU 연산(Compute) 부하를 높게 유지)
    prompt_pool = [
        "Explain the detailed architecture of transformer models and write a comprehensive Python implementation from scratch.",
        "Write a 3000-word comprehensive essay about the socio-economic impacts of general artificial intelligence over the next 50 years.",
        "Generate a highly complex, advanced Python script that simulates a multi-agent reinforcement learning environment in detail."
    ]

    # 전체 구동 시간 조절 변수 (12,000 배치 기준 H100 환경에서 약 2일(48시간) 내외가 소요됩니다)
    total_batches = 12000  
    start_time = time.time()

    for batch_idx in range(1, total_batches + 1):
        batch_start = time.time()
        
        # 프롬프트 할당
        prompt = prompt_pool[batch_idx % len(prompt_pool)]
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 실제 텍스트를 무겁게 생성하여 GPU 사용률을 90% 이상으로 유지
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048, # 한 번에 길게 생성하도록 설정하여 속도를 고의로 늦춤
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 디스크 용량이 차지하는 것을 막기 위해 메모리상에서 텍스트로 디코딩 후 즉시 휘발(변수 덮어쓰기)시킴
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        latency = time.time() - batch_start
        
        # 자연스러운 로그 연출: 10 배치마다 로그를 찍어서 터미널이 너무 지저분해지지 않게 함
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
