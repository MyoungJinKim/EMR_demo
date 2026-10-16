# NLP&AI Voice → SOAP Demo

한국어 진료 음성을 전사하고 **한국어·영어 SOAP 노트**를 생성하는 로컬 모델 데모입니다. 브라우저 녹음과 음성 파일 업로드, 전사문 편집, 숫자 구간 재생, 한영 노트 복사·다운로드를 지원합니다.

```text
브라우저 녹음/파일
  → 16 kHz 모노 변환 → 음성 구간 검출 → Whisper 전사
  → 전사문 확인/수정 → Qwen 영어 SOAP → 원문을 함께 참조한 한국어 SOAP
```

## 실행

Apple Silicon Mac에서 Python **3.13.0**으로 검증했습니다. 최초 설치와 모델 다운로드에는 인터넷 및 여러 GB의 저장 공간이 필요합니다. 준비한 모델로 로컬 추론할 때 유료 API 키는 필요하지 않습니다.

```bash
git clone https://github.com/MyoungJinKim/EMR_demo.git
cd EMR_demo
python3.13 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
python scripts/prepare_voice_demo.py
bash scripts/run_voice_demo.sh
```

브라우저에서 <http://127.0.0.1:8765/>를 열고 모델 준비 후 음성을 녹음하거나 업로드합니다. 준비 스크립트가 출력하는 `STT_MODEL`, `STT_DEVICE`, `EMR_MODEL_PATH` 환경변수는 다른 모델/경로를 선택할 때 적용합니다. 기본 모델은 `.demo-cache/`에서 자동으로 찾습니다. 첫 추론은 모델 로딩 때문에 더 오래 걸립니다.

이 저장소는 실행 스크립트가 `src/`를 Python 경로에 추가하는 방식입니다. `pip install .` 대신 위의 requirements 설치와 실행 스크립트를 사용합니다. 기존 인증 평가용 학습·평가 환경은 포함하지 않았습니다.

## 모델 및 처리 방식

| 단계 | Apple Silicon 기본값 | 설정 |
| --- | --- | --- |
| STT | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | MLX FP16, temperature 0, 이전 발화 문맥 비활성화 |
| 음성 전처리 | PyAV + faster-whisper의 Silero VAD | 16 kHz 모노, 최대 28초 구간, 무음 2초, 구간 여백 400ms |
| SOAP 생성 | [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | MLX 4bit, thinking 비활성화, greedy 생성 |
| 한영 일치 | 영어 SOAP 생성 후 원문·영어 노트로 한국어 생성 | 숫자 검증, SOAP 형식 정리, 제한적인 분류·한국어 문체 보정 |

모델 다운로드 revision은 코드에 고정되어 있습니다.

- MLX Whisper: `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`
- MLX Qwen: `8b2b98c00a6b4d291155e4890773ca8f769aee53`
- 일반 Qwen: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`

Apple Silicon 외 환경에서는 faster-whisper `large-v3`와 Transformers `Qwen/Qwen3.5-9B`를 선택합니다. CPU STT는 int8, CUDA STT는 float16을 사용합니다. 이 대체 경로의 실행 속도와 메모리 요구량은 장비에 따라 다르며, 이번 배포 확인은 Mac의 기본 MLX 경로를 대상으로 했습니다.

한국어 STT에는 일반 진료 어휘와 `확인한 뒤`를 힌트로 제공합니다. 용량·부정 표현을 정답으로 주입하거나 `귀`를 일괄적으로 `뒤`로 치환하지 않습니다. SOAP 보정은 명시적인 문장 패턴에만 적용되며 임의의 진단·용량을 만들어 채우지 않도록 설계했습니다. 단, 모델 출력의 의학적 정확성을 보장하지 않습니다.

## 사용 범위

- 음성: WAV, MP3, M4A, WebM, OGG, FLAC, MP4 / 최대 25 MB·10분.
- 전사문: 최대 20,000자. 동시 추론은 1건이며 다른 요청에는 409를 반환합니다.
- 자동 화자 분리 기능은 없습니다. 전사문에서 의사·환자 발언을 확인한 후 생성하세요.
- 출력은 검토용 초안입니다. 약물 용량, 부정 표현, 발언 주체와 SOAP 분류를 원음·원문과 대조해야 합니다. 개발용 합성 예제 검증은 독립적인 임상 정확도 평가나 의료기기 인증을 뜻하지 않습니다.

## 테스트

```bash
source .venv-demo/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
node tests/test_demo_ui.cjs
node --test tests/test_demo_worker.mjs
```

Python 회귀 검사는 API·입력 제한·인증·숫자/부정 표현·SOAP 분류·문체·모델 설정·서비스/터널 설정을 다룹니다. Node 검사는 모의 브라우저 UI 흐름과 Worker의 요청 전달을 확인합니다. 이 검사들은 모델 가중치를 다운로드하거나 실제 음성 추론을 실행하지 않습니다. 실제 녹음 파일과 과거 모델 비교 결과는 저장소에 포함하지 않았습니다.

## 외부 접속과 자동 실행

[DEMO.md](DEMO.md)에 비밀번호 설정, Mac 자동 실행, Cloudflare 고정 URL 구성 방법을 설명합니다. 고정 주소는 Worker와 named tunnel/VPC Service를 연결하는 구조이며, 실제 추론은 Mac에서 실행됩니다. Mac이 로그인되어 있고 깨어 있으며 인터넷에 연결되어 있어야 합니다.

## 구성 및 출처

- `src/mediqa_qwen_eval/`: API, STT, SOAP 생성·검증, 브라우저 UI와 NLP&AI 로고.
- `scripts/`: 모델 준비, 실행, macOS 서비스와 터널 관리.
- `deploy/cloudflare/`: Worker 소스와 계정별로 채우는 설정 예시.
- `tests/`: 회귀 검사 및 출처가 기재된 MTS-Dialog 참고 자료.

`tests/fixtures/mts_dialog_soap_train.json`은 [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) **train split 8개 예제**를 참조합니다. 원문은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)이며, 저자·논문·원본 revision·변경 사항은 파일의 `provenance`에 기재했습니다. 한국어 번역과 SOAP 점검 항목은 이 데모를 위해 작성한 자료이며 공식 한국어 정답이나 공식 SOAP 라벨이 아닙니다. 참고 논문: [Ben Abacha et al., EACL 2023](https://aclanthology.org/2023.eacl-main.168/).

모델 가중치와 전체 데이터셋은 배포하지 않습니다. 모델 사용 조건은 각 모델 저장소를 확인하세요. `.demo-tools/`, `.demo-cache/`, `output/` 등에는 로그인·터널 토큰·비밀번호·녹음·생성 결과가 들어갈 수 있으므로 Git에서 제외합니다. 실제 계정 정보 없이 실행·배포 소스와 설정 예시만 보관합니다.
