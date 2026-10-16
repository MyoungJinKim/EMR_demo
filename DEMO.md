# 실행과 외부 접속

모든 명령은 저장소 루트에서 실행한다. 이 저장소에는 실행 코드와 테스트만 있으며, 모델 가중치·음원·생성 노트·운영 계정 설정은 포함하지 않는다. 기존 운영 폴더와 별도로 clone한 경우 서비스와 자격 증명도 새로 준비해야 한다.

## 로컬 실행

Apple Silicon Mac에서는 Whisper large-v3-turbo MLX FP16과 Qwen3.5-9B MLX 4bit를 사용한다. 모델 가중치만 약 7.6GB이며, 16GB Mac에서는 두 모델을 순서대로 메모리에 올린다. 다른 장비의 기본 경로는 faster-whisper large-v3와 Transformers Qwen3.5-9B다. 해당 경로는 더 많은 메모리가 필요하며, 이 Mac과 같은 품질·속도를 검증한 것은 아니다.

검증 환경과 같은 Python 3.13으로 별도 환경을 만든다. 기존 인증 평가 환경과 의존성을 섞지 않는다.

```bash
python3.13 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python scripts/prepare_voice_demo.py
bash scripts/run_voice_demo.sh
```

[로컬 화면](http://127.0.0.1:8765/)에서 녹음하거나 음성 파일을 업로드한다. 준비 스크립트는 모델을 `.demo-cache/`에 다운로드한다. 런처가 `PYTHONPATH`를 설정하므로 `pip install .`은 필요 없다. 추론 중에는 모델을 다운로드하지 않으며, 첫 요청에는 모델 로딩 시간이 포함된다.

이전에 지정한 `STT_MODEL`, `STT_DEVICE`, `EMR_MODEL`, `EMR_MODEL_PATH`, `EMR_DEVICE`, `EMR_BASE_URL`, `EMR_ADAPTER_PATH` 환경변수는 기본값보다 우선한다. 처음 재현할 때는 해당 변수가 없는 새 터미널을 사용한다. 실행 중인 서버가 8765 포트를 사용한다면 별도 로컬 확인에는 `bash scripts/run_voice_demo.sh --port 8766`을 사용한다. 외부 접속 구성은 8765 기준이다.

## 인증을 켠 로컬 실행

기본 로컬 실행은 인증 없이 loopback 주소에서만 제공된다. 외부 공개 전에는 인증 설정을 생성한다.

```bash
.venv-demo/bin/python scripts/voice_demo_service.py prepare
export DEMO_ACCESS_FILE="$PWD/output/demo/service/access.json"
export DEMO_PUBLIC_URL_FILE="$PWD/output/demo/service/public-url.txt"
bash scripts/run_voice_demo.sh
```

`prepare`는 서버나 터널을 시작하지 않는다. `output/demo/service/access.json`에 계정 `demo`와 임의의 비밀번호를 생성하며, 기존 정상 설정이 있으면 유지한다. 비밀번호는 이 로컬 파일에서 확인한다. 폴더 권한은 0700, 인증 파일은 0600이다. 파일·비밀번호를 Git에 추가하거나 접속 URL에 넣지 않는다.

인증 모드는 페이지·정적 파일·API를 모두 보호한다. POST 요청은 로컬 화면 또는 `public-url.txt`에 기록된 정확한 HTTPS Origin만 허용한다. 해당 파일은 터널 실행기가 관리한다. 인증 설정을 바꾸면 서버를 재시작해야 한다.

## 선택: 도메인 구매 없는 고정 HTTPS 주소

연결 구조는 `workers.dev` 주소 → Cloudflare Worker → 단일 VPC Service → Cloudflare Tunnel → `127.0.0.1:8765`다. 모델은 Mac에서 실행되지만 외부 기기의 음성·전사문·SOAP는 Cloudflare 중계망을 통과한다. 동의받은 내용이나 가상 진료 대화로 시연한다.

이 저장소의 `deploy/cloudflare/worker.mjs`와 `wrangler.example.jsonc`를 사용한다. 계정·터널·VPC Service는 각자의 Cloudflare 계정에 별도로 생성해야 한다. 운영 중인 다른 데모의 리소스 ID나 토큰을 복사하지 않는다.

### 도구 준비

Cloudflare 계정과 지원되는 Node.js LTS가 필요하다. [Wrangler 설치 안내](https://developers.cloudflare.com/workers/wrangler/install-and-update/)와 [cloudflared 다운로드 안내](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)를 따른다. 기존 데모에서 확인한 버전은 Wrangler 4.151.1, cloudflared 2026.10.0이다.

- Wrangler를 프로젝트의 `.demo-tools/cloudflare-cli/` 안에 npm 패키지로 설치한다. `scripts/demo_cloudflare.sh`는 `.demo-tools/cloudflare-cli/node_modules/wrangler/bin/wrangler.js`를 실행한다.
- OS·아키텍처에 맞는 cloudflared 실행 파일을 `.demo-tools/cloudflared`에 두고 실행 권한을 부여한다. 터널 실행기는 Homebrew의 `/opt/homebrew/bin/cloudflared`, `/usr/local/bin/cloudflared`도 찾지만, Cloudflare CLI 래퍼는 프로젝트 내부 경로를 사용한다.
- 래퍼는 로그인·로그를 비공개 `.demo-tools/cloudflare-config/`에 저장한다. 이 폴더 전체를 공유하지 않는다.

```bash
npm install --prefix .demo-tools/cloudflare-cli --save-exact wrangler@4.151.1
bash scripts/demo_cloudflare.sh login --scopes account:read user:read workers_scripts:write connectivity:admin
```

위 로그인은 계정 조회, Worker 배포, Tunnel/VPC 연결 관리에 사용할 권한을 요청한다. 브라우저의 Cloudflare 로그인과 Wrangler 권한 승인을 완료한다. [공식 VPC 시작 안내](https://developers.cloudflare.com/workers-vpc/get-started/)의 계정별 설정을 수행하되 다음 값을 사용한다.

1. 이 Mac에서 실행할 계정 등록 Tunnel을 만든다. 자동 실행은 아래 프로젝트 스크립트로 관리하므로 별도의 시스템 cloudflared 서비스를 중복 설치하지 않는다.
2. 그 Tunnel에 연결되는 HTTP VPC Service를 만든다. 대상 IP는 `127.0.0.1`, HTTP 포트는 `8765`다. Worker에는 이 서비스 하나만 바인딩한다.
3. `deploy/cloudflare/wrangler.example.jsonc`를 `output/demo/service/wrangler.jsonc`로 복사한다. `main`은 `../../../deploy/cloudflare/worker.mjs`, `service_id`는 방금 생성한 서비스 ID로 변경한다. Worker 이름과 계정은 자신의 설정을 사용한다. 바인딩 이름 `DEMO_SERVICE`, `workers_dev: true`, 관측 로그·미리보기 URL 비활성화 설정은 유지한다.
4. 계정의 workers.dev 하위 도메인을 설정하고 아래 명령으로 Worker를 배포한다. 배포된 URL을 다음 단계에 사용한다.

```bash
bash scripts/demo_cloudflare.sh deploy --config output/demo/service/wrangler.jsonc
```

### 터널 설정

`prepare`로 만든 비공개 폴더에 다음 파일을 로컬 편집기로 작성한다. 예시 주소는 자신의 실제 배포 URL로 바꾼다.

`output/demo/service/fixed-tunnel.json`:

```json
{"public_url": "https://your-worker.your-subdomain.workers.dev"}
```

`output/demo/service/tunnel-token`에는 해당 Tunnel의 연결 토큰만 저장한다. JSON이나 따옴표로 감싸지 않는다. 토큰을 명령 인자·문서·소스 코드에 붙여 넣지 않는다.

```bash
chmod 600 output/demo/service/fixed-tunnel.json output/demo/service/tunnel-token
```

URL은 끝의 `/`나 경로 없이 위 형태의 HTTPS 주소여야 한다. 터널 실행기는 `--token-file`로 토큰을 읽는다. 잘못된 고정 설정은 임시 주소로 대체하지 않고 실패한다. Workers VPC에는 QUIC와 외부 방향 UDP 7844 연결이 필요하다. [공식 Tunnel 요구사항](https://developers.cloudflare.com/workers-vpc/configuration/tunnel/).

## macOS 로그인 자동 실행

먼저 모델·cloudflared·인증 파일을 준비하고, 고정 주소를 쓰려면 위 설정도 완료한다. 수동 실행한 8765 서버는 종료한 뒤 설치한다. 설치기는 기존 프로세스를 강제 종료하지 않는다.

```bash
.venv-demo/bin/python scripts/voice_demo_service.py install
.venv-demo/bin/python scripts/voice_demo_service.py status
```

`org.nlpai.voice-demo`와 `org.nlpai.voice-demo.tunnel` 사용자 LaunchAgent가 서버와 터널을 관리한다. 설치 시 바로 실행되며, 종료되면 다시 시작한다. 같은 Mac에서 다른 clone이 이미 같은 이름으로 등록되어 있으면 덮어쓰지 않는다. 이 경우 기존 운영 위치를 확인하고 그 위치에서 먼저 `uninstall`해야 한다.

고정 설정이 없으면 무료 Quick Tunnel을 시작하며 주소가 재시작 때마다 바뀐다. 고정 설정이 있으면 Worker와 계정의 주소를 유지하는 한 터널 재시작 뒤에도 같은 주소를 사용한다. 현재 주소는 `output/demo/service/public-url.txt`에서 확인한다. 로그는 같은 폴더의 `server.*.log`, `tunnel.*.log`에 남는다.

```bash
.venv-demo/bin/python scripts/voice_demo_service.py uninstall
```

해제는 자동 실행만 제거한다. 코드·모델·비밀번호·로그와 Cloudflare의 Worker/Tunnel/VPC 리소스는 유지된다. 외부 공개를 완전히 종료하려면 Cloudflare 계정에서도 해당 리소스를 정리한다.

## 연결 확인과 운영 조건

- 외부 주소에서 인증 없는 요청이 401인지 확인한다. `demo`와 로컬 비밀번호로 로그인한 뒤 가상 음원을 업로드하고 전사·한국어·영어 SOAP가 표시되는지 확인한다.
- 터널을 다시 시작한 뒤 동일 URL이 유지되는지 확인한다. 주소 파일이 남아 있어도 Mac과 터널이 꺼져 있으면 요청은 실패할 수 있다.
- Mac이 로그인된 채 깨어 있고 인터넷에 연결되어 있어야 한다. 터미널이나 Codex 종료와 서비스 수명은 별개지만, Mac 종료·절전·로그아웃 중에는 접속할 수 없다.
- 한 번에 하나의 추론을 처리한다. 모델 준비 표시가 두 모델의 동시 GPU 상주를 뜻하지 않는다. 지원 입력은 최대 25MB·10분이다.
- 2026-10-16 확인 기준 Workers VPC는 공개 베타 기간에 무료이며, Worker 요청·계산에는 Workers 요금제가 적용된다. 무료 유지를 원하면 Free 요금제 한도를 확인한다. 베타 이후 무료를 보장하지 않는다. [VPC 가격](https://developers.cloudflare.com/workers-vpc/platform/pricing/), [Workers 가격](https://developers.cloudflare.com/workers/platform/pricing/).

코드 테스트는 README의 명령으로 실행한다. 이 저장소는 기존 개발 음원·개별 생성 결과·운영 계정 설정을 배포하지 않는다. 실제 마이크·다른 장비·실제 Mac 재부팅은 각 배포 환경에서 별도로 확인해야 한다. 생성 노트는 검토용 초안이며, 원음과 대조해 약명·용량·부정 표현·SOAP 분류를 확인한다.
