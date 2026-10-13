# CrossFit Report LLM

크로스핏 동작 분석에서 생성한 모델 평가 PDF를 읽고, 검색 기반 문맥 구성과 LLM·VLM을 이용해 분석 보고서를 작성하는 Python 프로젝트입니다.

## 처리 과정

1. **PDF 처리**: 페이지별 텍스트를 추출하고 페이지 이미지를 생성합니다.
2. **시각 정보 추출**: Qwen2.5-VL로 성능표와 그래프의 정보를 읽습니다.
3. **페이지 검색**: 두 MiniLM 인코더의 평균 임베딩과 코사인 유사도로 관련 페이지를 찾습니다.
4. **문맥 구성**: 검색 결과에 성능표·혼동행렬·ROC·PR 등 유형별 페이지를 보강합니다.
5. **보고서 생성**: 문맥을 텍스트 생성 모델에 전달하고 분석 결과를 파일로 저장합니다.

`train_gici.py`는 검색용 인코더를 반복 학습하고 파라미터를 평균 병합·동기화하는 실험 코드입니다. 생성 모델과 검색용 인코더는 별도로 구성됩니다.

## 코드 구성

| 경로 | 역할 |
|---|---|
| `train_gici.py` | 검색용 인코더 학습 및 저장 |
| `inference_gici_modular.py` | 보고서 생성 실행부 |
| `retrievers/gici_retriever.py` | 페이지 임베딩 및 유사도 검색 |
| `pdf_utils/page_loader.py` | PDF 텍스트 추출 및 이미지 렌더링 |
| `model_backends/` | 텍스트 생성·시각 정보 추출 백엔드 |
| `report_context_utils.py` | 페이지 유형 분류·선택·문맥 구성 |
| `report_prompt_utils.py` | 보고서 생성 프롬프트 |

## 설치

프로젝트 루트에서 가상환경을 만들고 의존성을 설치합니다. PyTorch는 사용하는 CPU·CUDA 환경에 맞게 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

기본 Qwen 백엔드는 `.env`의 `HF_TOKEN`을 사용합니다. 개인 토큰은 로컬 `.env`에만 입력합니다.

## 사용 방법

### 1. 검색용 인코더 준비

학습용 텍스트를 다음 경로에 둡니다. 각 `.txt` 파일을 하나의 학습 샘플로 읽으며 최대 길이를 초과하는 내용은 잘립니다.

```text
data/
├── korean/*.txt
└── english/*.txt
```

`train_gici.py`의 `parse_args()` 안에 있는 `config_dict`에서 데이터 경로, 모델, 반복 횟수와 배치 크기를 설정한 뒤 실행합니다.

```bash
python train_gici.py
```

기본 저장 위치는 `saved_models_gici/`입니다. 추론에는 Hugging Face 형식으로 저장한 `final_mkor/`, `final_meng/` 폴더를 사용합니다. 준비된 인코더가 있다면 추론 설정에서 해당 경로를 지정할 수 있습니다.

### 2. 평가 PDF 준비

분석할 파일은 다음 이름 패턴으로 `reports/`에 둡니다.

```text
reports/
└── example_Model_Evaluation_Report.pdf
```

### 3. 보고서 생성

`inference_gici_modular.py` 하단의 `GICIInferenceConfig`에서 인코더 경로와 생성 모델을 설정합니다.

- `mkor_dir`, `meng_dir`: 검색용 인코더 경로
- `llm_provider`, `llm_model`: 텍스트 생성 백엔드와 모델
- `use_vlm`, `vlm_model`: 시각 정보 추출 사용 여부와 모델
- `max_context_tokens`, `max_new_tokens`: 문맥·출력 길이
- `folder_path`: 입력 PDF 폴더

기본 실행 설정은 Qwen2.5-3B-Instruct와 Qwen2.5-VL-7B-Instruct를 사용합니다. 모델 가중치를 내려받을 공간과 두 모델을 구동할 메모리를 준비합니다.

```bash
python inference_gici_modular.py
```

결과는 `outputs/*_analysis.txt`에, 페이지별 텍스트·시각 정보는 `outputs/page_level_extractions/`에 저장됩니다.

### 백엔드 변경

| 설정 | 추가 준비 |
|---|---|
| `llm_provider="hf_qwen"` | `HF_TOKEN` 및 Qwen 모델 |
| `llm_provider="openai"` | 로컬 `OPENAI_API_KEY`, 사용할 모델명을 `llm_model`에 지정 |
| `llm_provider="ollama"` | 로컬 Ollama 서버와 `ollama_model`에 지정한 모델 |

OpenAI 백엔드를 선택하면 구성한 보고서 문맥이 해당 API로 전송됩니다. 텍스트만 사용할 때는 `use_vlm=False`로 설정합니다.

입력 문서·학습 데이터·모델 가중치·생성 파일은 로컬에서 준비하고 관리합니다.
