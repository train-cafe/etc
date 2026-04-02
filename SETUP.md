# 환경 구성 가이드

## 개요

이 레포지토리에는 두 개의 Python 스크립트가 포함되어 있습니다.

| 스크립트 | 설명 | 사용 모델 |
|---|---|---|
| `data_generation.py` | NSFW 차단 모델 학습을 위한 데이터 생성 코드 | `huihui-ai/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated` |
| `synthetic_data_pipeline.py` | 대규모 합성 데이터 증강 파이프라인 | `Qwen/Qwen2.5-72B-Instruct` |

---

## 하드웨어 요구사항

- **GPU**: NVIDIA H100 80GB HBM3 x 2장 이상
  - `data_generation.py`: 2 GPU 사용 (tensor parallelism)
  - `synthetic_data_pipeline.py`: 2 GPU 사용 (`CUDA_VISIBLE_DEVICES="2,3"` 기본 설정, 필요시 수정)
- **VRAM**: GPU당 80GB (72B 모델 bfloat16 로딩 시 약 144GB 필요)
- **디스크**: 모델 가중치 캐시를 위한 충분한 저장 공간 (모델당 수십~수백 GB)

---

## 소프트웨어 요구사항

### 기본 환경

- **Python**: 3.10+
- **CUDA**: 12.2+
- **cuDNN**: 9.1+
- **PyTorch**: 2.6.0+cu124

### 현재 환경에서 추가 설치가 필요한 패키지

현재 환경(`pip list` 기준)에 이미 설치되어 있는 항목: `torch`, `numpy`, `pillow`, `requests` 등

아래 패키지를 **추가로 설치**해야 합니다:

```bash
pip install vllm tqdm transformers
```

#### 패키지별 설명

| 패키지 | 사용처 | 설명 |
|---|---|---|
| `vllm` | `data_generation.py` | 고성능 LLM 추론 엔진 (tensor parallelism 지원) |
| `tqdm` | `data_generation.py` | 진행률 표시 바 |
| `transformers` | `synthetic_data_pipeline.py` | HuggingFace 모델 로딩 (`AutoModelForCausalLM`, `AutoTokenizer`) |

> **참고**: `vllm` 설치 시 호환되는 PyTorch 및 CUDA 버전이 자동으로 확인됩니다. 현재 환경(PyTorch 2.6.0+cu124, CUDA 12.2)에서 정상 동작합니다.

---

## 설치 및 실행

### 1. 패키지 설치

```bash
pip install vllm tqdm transformers
```

### 2. 디렉토리 구조 확인

```
etc/
├── data_generation.py
├── synthetic_data_pipeline.py
└── SETUP.md
```

### 3. 경로 설정 (필요시 수정)

#### `data_generation.py`

```python
OUTPUT_FILE = "/user-volume/nsfw_prompt_dataset_high_t_safe.json"  # 출력 파일 경로
```

- `/user-volume/` 디렉토리가 존재하는지 확인하고, 없으면 경로를 수정하세요.

#### `synthetic_data_pipeline.py`

```python
CACHE_DIR = "/group-volume/kang9.lee/models"  # 모델 캐시 경로
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"    # 사용할 GPU 인덱스
```

- `CACHE_DIR`: 모델 가중치가 저장될 경로입니다. 접근 가능한 경로로 수정하세요.
- `CUDA_VISIBLE_DEVICES`: 사용할 GPU 번호를 환경에 맞게 수정하세요 (예: `"0,1"`).

### 4. 실행

```bash
# NSFW 차단 모델용 데이터 생성
python data_generation.py

# 합성 데이터 증강 파이프라인
python synthetic_data_pipeline.py
```

---

## 참고 사항

- `data_generation.py`는 NSFW 콘텐츠를 탐지/차단하는 모델을 만들기 위한 **학습 데이터 생성** 코드입니다.
- `synthetic_data_pipeline.py`는 12,000 배치를 처리하며, 예상 소요 시간은 약 **48시간**입니다.
- HuggingFace 모델을 처음 실행할 때 자동으로 다운로드됩니다. 네트워크 환경과 모델 크기에 따라 상당한 시간이 소요될 수 있습니다.
- HuggingFace에서 gated 모델을 사용하는 경우 `huggingface-cli login`으로 인증이 필요할 수 있습니다.
