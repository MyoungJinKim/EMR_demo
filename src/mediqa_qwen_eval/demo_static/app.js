"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const transcript = $("transcript");
  const canvas = $("waveform");
  const context = canvas.getContext("2d");
  const limits = { audio_mb: 25, audio_seconds: 600, text_chars: 20000 };
  let busy = "";
  let sequence = 0;
  let controller = null;
  let recorder = null;
  let stream = null;
  let audioContext = null;
  let analyser = null;
  let waveformFrame = null;
  let clock = null;
  let recordingStarted = 0;
  let audioUrl = null;
  let reviewEnd = null;
  let notes = null;
  let noteLanguage = "ko";
  let ready = false;

  const formatTime = (seconds) => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;

  function setStatus(message) {
    $("status-text").textContent = message;
  }

  function showError(error) {
    $("error").textContent = error.message || "처리 중 오류가 발생했습니다. 다시 시도해 주세요.";
    $("error").hidden = false;
  }

  function clearError() {
    $("error").hidden = true;
    $("error").textContent = "";
  }

  function updateControls() {
    const working = Boolean(busy);
    $("record").disabled = working && busy !== "recording";
    $("record-label").textContent = busy === "recording" ? "녹음 종료 · 생성" : busy === "permission" ? "마이크 연결 중" : "녹음 시작";
    for (const id of ["upload", "example", "input-language", "warmup"]) $(id).disabled = working;
    $("warmup").textContent = busy === "warmup" ? "모델 준비 중…" : ready ? "모델 준비 완료 ✓" : "모델 미리 준비 ↗";
    $("warmup").disabled = working || ready;
    $("cancel").hidden = !working;
    $("cancel").textContent = busy === "recording" ? "녹음 취소" : "작업 취소";
    $("generate").disabled = working || !transcript.value.trim() || transcript.value.length > limits.text_chars;
    $("generate-label").textContent = busy === "generating" ? "두 언어 생성 중…" : notes ? "다시 생성" : "한·영 SOAP 생성";
    $("reset").disabled = working || !(transcript.value || audioUrl || notes);
    transcript.readOnly = working;
    $("copy").disabled = working || !notes;
    $("download").disabled = working || !notes;
    $("char-count").textContent = `${transcript.value.length.toLocaleString("ko-KR")} / ${limits.text_chars.toLocaleString("ko-KR")}자`;
    document.body.classList.toggle("recording", busy === "recording");
    document.body.classList.toggle("busy", working && busy !== "recording" && busy !== "permission");
    document.body.classList.toggle("has-note", Boolean(notes));
    $("workspace").setAttribute("aria-busy", String(working));
  }

  function setPipeline(stage) {
    ["record", "transcribe", "generate"].forEach((name, index) => {
      const step = $(`step-${name}`);
      step.classList.toggle("current", index === stage);
      step.classList.toggle("complete", index < stage);
      if (index === stage) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
  }

  function setBusy(value, message) {
    busy = value;
    clearInterval(clock);
    clock = null;
    $("process-time").hidden = !["transcribing", "generating", "warmup"].includes(value);
    if (!$("process-time").hidden) {
      const started = performance.now();
      const tick = () => { $("process-time").textContent = `${formatTime((performance.now() - started) / 1000)} 경과`; };
      tick();
      clock = setInterval(tick, 1000);
    }
    if (message) setStatus(message);
    updateControls();
  }

  function emptyNote(title = "대화를 SOAP 기록으로 정리합니다.", description = "주관적 정보 · 객관적 소견 · 평가 · 계획을 한국어와 영어로 정리합니다.") {
    notes = null;
    $("note").textContent = "";
    $("note").hidden = true;
    $("note-empty").hidden = false;
    $("note-empty-title").textContent = title;
    $("note-empty-description").textContent = description;
    $("note-meta").textContent = "";
    $("bilingual-status").textContent = "두 언어 동시 생성";
    $("warnings").hidden = true;
    updateControls();
  }

  function selectNote(language, focus = false) {
    noteLanguage = language;
    for (const lang of ["ko", "en"]) {
      $(`tab-${lang}`).setAttribute("aria-selected", String(lang === language));
      $(`tab-${lang}`).tabIndex = lang === language ? 0 : -1;
    }
    $("note-panel").setAttribute("aria-labelledby", `tab-${language}`);
    $("note").lang = language;
    if (notes) {
      const parts = notes[language].split(/^(S · (?:주관적 정보|Subjective)|O · (?:객관적 소견|Objective)|A · (?:평가|Assessment)|P · (?:계획|Plan))$/m);
      $("note").replaceChildren(...parts.map((part, index) => {
        if (index % 2 === 0) return document.createTextNode(part);
        const heading = document.createElement("strong");
        heading.className = "soap-heading";
        heading.setAttribute("role", "heading");
        heading.setAttribute("aria-level", "3");
        heading.textContent = part;
        return heading;
      }));
    }
    if (focus) $(`tab-${language}`).focus();
  }

  async function api(path, options = {}, signal) {
    let response;
    try {
      response = await fetch(path, { ...options, signal });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error("서버에 연결할 수 없습니다. 데모 서버가 실행 중인지 확인해 주세요.");
    }
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`서버 응답을 읽을 수 없습니다 (HTTP ${response.status}).`); }
    if (!response.ok) {
      const detail = data.detail || data.error;
      const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((item) => item.msg).join(" · ") : detail?.message;
      throw new Error(message || `요청을 처리하지 못했습니다 (HTTP ${response.status}).`);
    }
    return data;
  }

  function showHealth(data) {
    ready = typeof data.ready === "boolean" ? data.ready : Boolean(data.stt?.loaded && data.emr?.loaded);
    $("health-dot").className = "health-dot ready";
    $("health-label").textContent = ready ? "모델 준비 완료" : "서버 연결됨";
    const local = ["local", "transformers", "mlx"].includes(data.emr?.backend);
    $("model-info").textContent = `STT · ${data.stt?.model || "설정 확인 필요"}   /   EMR · ${data.emr?.model || "설정 확인 필요"} (${local ? "로컬" : data.emr?.backend || "백엔드 미확인"})`;
    $("data-notice").textContent = local ? "음성 인식과 EMR 모델은 데모 서버에서 실행됩니다. 외부 주소로 접속하면 음성·텍스트가 HTTPS 터널을 통해 전송됩니다. 동의받은 내용 또는 가상 진료 내용을 사용하세요." : "음성은 데모 서버에서 전사되며, 전사 텍스트는 설정된 EMR 백엔드로 전송됩니다. 동의받은 내용 또는 가상 진료 내용을 사용하세요.";
    if (data.limits) {
      for (const key of Object.keys(limits)) {
        if (Number.isFinite(data.limits[key]) && data.limits[key] > 0) limits[key] = data.limits[key];
      }
      transcript.maxLength = limits.text_chars;
    }
    updateControls();
  }

  async function refreshHealth() {
    try { showHealth(await api("/api/health")); }
    catch {
      ready = false;
      $("health-dot").className = "health-dot failed";
      $("health-label").textContent = "서버 연결 확인 필요";
      $("model-info").textContent = "모델 정보를 확인할 수 없습니다. 서버 실행 상태를 확인해 주세요.";
      updateControls();
    }
  }

  function drawWaveform() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const values = analyser ? new Uint8Array(analyser.frequencyBinCount) : null;
    if (values) analyser.getByteTimeDomainData(values);
    const count = Math.floor(width / 6);
    context.strokeStyle = analyser ? "#666666" : "#d9d9d9";
    context.lineWidth = 2;
    context.lineCap = "round";
    for (let i = 0; i < count; i++) {
      const level = values ? Math.abs(values[Math.floor(i * values.length / count)] - 128) / 128 : 0;
      const barHeight = Math.max(2, Math.min(height - 10, level * height * 2.2));
      const x = (i + 0.5) * width / count;
      context.beginPath();
      context.moveTo(x, (height - barHeight) / 2);
      context.lineTo(x, (height + barHeight) / 2);
      context.stroke();
    }
    if (analyser) waveformFrame = requestAnimationFrame(drawWaveform);
  }

  function releaseMicrophone() {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    if (audioContext && audioContext.state !== "closed") audioContext.close().catch(() => {});
    audioContext = null;
    analyser = null;
    cancelAnimationFrame(waveformFrame);
    waveformFrame = null;
    drawWaveform();
  }

  function clearNumericReview() {
    if (reviewEnd !== null) $("audio-player").pause();
    reviewEnd = null;
    $("numeric-segments").replaceChildren();
    $("numeric-review").hidden = true;
    $("numeric-review").open = false;
  }

  function showNumericReview(segments) {
    clearNumericReview();
    if (!audioUrl || !Array.isArray(segments)) return;
    const sourceUrl = audioUrl;
    for (const segment of segments) {
      if (typeof segment?.text !== "string" || !/\d/.test(segment.text) || !Number.isFinite(segment.start) || !Number.isFinite(segment.end) || segment.start < 0 || segment.end <= segment.start) continue;
      const button = document.createElement("button");
      button.className = "segment-button";
      button.type = "button";
      button.textContent = `${formatTime(segment.start)} · ${segment.text.trim()}`;
      button.setAttribute("aria-controls", "audio-player");
      button.title = "이 구간 원음 재생";
      button.addEventListener("click", async () => {
        if (audioUrl !== sourceUrl || $("numeric-review").hidden) return;
        const player = $("audio-player");
        const end = segment.end + 0.4;
        try {
          player.currentTime = Math.max(0, segment.start - 0.3);
          reviewEnd = end;
          await player.play();
        } catch {
          if (audioUrl !== sourceUrl || reviewEnd !== end) return;
          reviewEnd = null;
          setStatus("원음 재생을 시작하지 못했습니다. 음성 플레이어에서 직접 재생해 주세요.");
        }
      });
      $("numeric-segments").appendChild(button);
    }
    $("numeric-review").hidden = !$("numeric-segments").children.length;
  }

  $("audio-player").addEventListener("timeupdate", () => {
    if (reviewEnd !== null && $("audio-player").currentTime >= reviewEnd) {
      reviewEnd = null;
      $("audio-player").pause();
    }
  });

  function removeAudio() {
    clearNumericReview();
    $("audio-player").pause();
    $("audio-player").removeAttribute("src");
    $("audio-player").load();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = null;
    $("audio-preview").hidden = true;
  }

  function previewAudio(blob, name) {
    removeAudio();
    audioUrl = URL.createObjectURL(blob);
    $("audio-player").src = audioUrl;
    $("audio-name").textContent = name;
    $("audio-preview").hidden = false;
  }

  function newOperation() {
    controller?.abort();
    controller = new AbortController();
    clearError();
    return ++sequence;
  }

  async function generateNotes(operation) {
    const text = transcript.value.trim();
    if (!text) throw new Error("인식된 텍스트가 없습니다. 음량을 확인하거나 대화를 직접 입력해 주세요.");
    if (text.length > limits.text_chars) throw new Error(`진료 대화는 ${limits.text_chars.toLocaleString("ko-KR")}자 이하로 입력해 주세요.`);
    emptyNote("두 언어로 SOAP 기록을 작성하고 있습니다.", "같은 진료 내용으로 두 언어의 SOAP 초안을 작성합니다. 첫 실행에는 모델 준비로 수 분이 걸릴 수 있습니다.");
    setPipeline(2);
    setBusy("generating", "한국어와 영어 SOAP 초안을 생성 중입니다. 첫 실행에는 수 분이 걸릴 수 있습니다.");
    const result = await api("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, language: "both" })
    }, controller.signal);
    if (operation !== sequence) return;
    if (typeof result.notes?.ko !== "string" || !result.notes.ko.trim() || typeof result.notes?.en !== "string" || !result.notes.en.trim()) {
      throw new Error("두 언어의 SOAP 초안을 모두 받지 못했습니다. 다시 생성해 주세요.");
    }
    notes = { ko: result.notes.ko.trim(), en: result.notes.en.trim() };
    selectNote(noteLanguage);
    $("note").hidden = false;
    $("note-empty").hidden = true;
    $("note-meta").textContent = Number.isFinite(result.elapsed_seconds) ? `${result.elapsed_seconds.toFixed(1)}초` : "";
    $("bilingual-status").textContent = "두 언어 생성 완료";
    const warnings = (result.warnings || []).filter((warning) => typeof warning === "string");
    $("warnings").textContent = warnings.join(" · ");
    $("warnings").hidden = !warnings.length;
    setPipeline(3);
    setBusy("", "한국어와 영어 SOAP 초안이 완성되었습니다. 원문과 비교하여 검토해 주세요.");
    refreshHealth();
  }

  async function processAudio(blob, name, operation) {
    try {
      if (blob.size === 0) throw new Error("녹음된 소리가 없습니다. 마이크를 확인한 뒤 다시 녹음해 주세요.");
      if (blob.size > limits.audio_mb * 1024 * 1024) throw new Error(`음성 파일은 ${limits.audio_mb} MB 이하로 올려 주세요.`);
      previewAudio(blob, name);
      setPipeline(1);
      setBusy("transcribing", "음성을 텍스트로 전사 중입니다. 첫 실행에는 음성 모델 준비가 필요합니다.");
      const result = await api(`/api/transcribe?language=${encodeURIComponent($("input-language").value)}`, {
        method: "POST", headers: { "Content-Type": blob.type || "application/octet-stream" }, body: blob
      }, controller.signal);
      if (operation !== sequence) return;
      transcript.value = typeof result.text === "string" ? result.text.trim() : "";
      const duration = Number.isFinite(result.duration_seconds) ? ` · ${formatTime(result.duration_seconds)}` : "";
      $("transcript-meta").textContent = `음성 전사${duration}`;
      showNumericReview(result.segments);
      updateControls();
      await generateNotes(operation);
    } catch (error) {
      if (operation !== sequence || error.name === "AbortError") return;
      const hasText = Boolean(transcript.value.trim());
      emptyNote("아직 초안이 완성되지 않았습니다.", hasText ? "전사 텍스트를 확인하고 다시 생성할 수 있습니다." : "다른 음성 파일로 시도하거나 대화를 직접 입력해 주세요.");
      setBusy("", hasText ? "전사 텍스트는 유지됩니다. 확인 후 다시 생성해 주세요." : "음성 처리를 완료하지 못했습니다.");
      showError(error);
    }
  }

  async function startRecording() {
    if (busy) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      showError(new Error("이 브라우저에서 녹음을 사용할 수 없습니다. localhost 또는 HTTPS에서 최신 Chrome·Safari로 접속하거나 음성 파일을 올려 주세요."));
      return;
    }
    const operation = newOperation();
    setBusy("permission", "브라우저의 마이크 사용 권한을 허용해 주세요.");
    try {
      const microphone = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
      if (operation !== sequence) { microphone.getTracks().forEach((track) => track.stop()); return; }
      stream = microphone;
      const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm", "audio/ogg;codecs=opus"].find((type) => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const chunks = [];
      const currentRecorder = recorder;
      currentRecorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
      currentRecorder.addEventListener("stop", () => {
        if (operation !== sequence) return;
        releaseMicrophone();
        recorder = null;
        $("recording-state").lastChild.textContent = " 녹음 완료";
        processAudio(new Blob(chunks, { type: currentRecorder.mimeType || "audio/webm" }), "마이크 녹음", operation);
      });
      currentRecorder.addEventListener("error", () => {
        if (operation !== sequence) return;
        cancelOperation();
        showError(new Error("마이크 녹음 중 오류가 발생했습니다. 연결을 확인해 주세요."));
      });
      removeAudio();
      transcript.value = "";
      $("transcript-meta").textContent = "녹음 중";
      emptyNote();
      setPipeline(0);
      currentRecorder.start(1000);
      setBusy("recording", "진료 대화를 말씀해 주세요. 녹음 종료 후 두 언어의 SOAP 초안을 자동 생성합니다.");
      $("recording-state").lastChild.textContent = " 녹음 중";
      recordingStarted = performance.now();
      tickRecording();
      clock = setInterval(tickRecording, 250);
      // Audio visualization is optional; a missing WebAudio implementation must not interrupt recording.
      try {
        const AudioContextType = window.AudioContext || window.webkitAudioContext;
        if (AudioContextType) {
          audioContext = new AudioContextType();
          await audioContext.resume();
          if (operation !== sequence || busy !== "recording") return;
          analyser = audioContext.createAnalyser();
          analyser.fftSize = 512;
          audioContext.createMediaStreamSource(stream).connect(analyser);
          drawWaveform();
        }
      } catch { /* Recording still works without the visualizer. */ }
    } catch (error) {
      if (operation !== sequence) return;
      releaseMicrophone();
      recorder = null;
      setBusy("", "마이크를 시작하지 못했습니다. 음성 파일을 업로드할 수도 있습니다.");
      const messages = {
        NotAllowedError: "마이크 접근이 허용되지 않았습니다. 브라우저 주소창의 마이크 권한을 확인해 주세요.",
        NotFoundError: "사용 가능한 마이크를 찾지 못했습니다. 마이크를 연결하거나 음성 파일을 올려 주세요.",
        NotReadableError: "마이크를 사용할 수 없습니다. 다른 앱의 마이크 사용 여부를 확인해 주세요."
      };
      showError(new Error(messages[error.name] || error.message));
    }
  }

  function tickRecording() {
    const seconds = (performance.now() - recordingStarted) / 1000;
    const total = document.createElement("span");
    total.textContent = ` / ${formatTime(limits.audio_seconds)}`;
    $("timer").replaceChildren(document.createTextNode(formatTime(Math.min(seconds, limits.audio_seconds))), total);
    if (seconds >= limits.audio_seconds) stopRecording();
  }

  function stopRecording() {
    if (busy !== "recording" || !recorder || recorder.state === "inactive") return;
    clearInterval(clock);
    setBusy("stopping", "녹음을 마무리하고 있습니다.");
    recorder.stop();
    releaseMicrophone();
  }

  function cancelOperation() {
    ++sequence;
    controller?.abort();
    controller = null;
    if (recorder?.state === "recording") recorder.stop();
    recorder = null;
    releaseMicrophone();
    const wasRecording = ["permission", "recording", "stopping"].includes(busy);
    setBusy("", wasRecording ? "녹음을 취소했습니다." : "화면 작업을 취소했습니다. 서버의 진행 중인 처리는 마무리될 수 있습니다.");
    $("recording-state").lastChild.textContent = " 녹음 준비";
    if (wasRecording && !transcript.value) $("transcript-meta").textContent = "입력 대기";
    if (!notes) emptyNote("작업을 다시 시작할 수 있습니다.", "전사 텍스트가 있다면 내용을 확인하고 EMR을 생성하세요.");
    setPipeline(notes ? 3 : transcript.value.trim() ? 2 : 0);
  }

  $("record").addEventListener("click", () => busy === "recording" ? stopRecording() : startRecording());
  $("cancel").addEventListener("click", cancelOperation);
  $("upload").addEventListener("click", () => $("audio-file").click());
  $("audio-file").addEventListener("change", () => {
    const file = $("audio-file").files[0];
    $("audio-file").value = "";
    if (!file || busy) return;
    clearNumericReview();
    if (file.size > limits.audio_mb * 1024 * 1024) { showError(new Error(`음성 파일은 ${limits.audio_mb} MB 이하로 올려 주세요.`)); return; }
    if (!file.size) { showError(new Error("빈 파일은 처리할 수 없습니다.")); return; }
    const operation = newOperation();
    transcript.value = "";
    $("transcript-meta").textContent = "파일 전사 대기";
    emptyNote();
    processAudio(file, file.name, operation);
  });
  transcript.addEventListener("input", () => {
    clearNumericReview();
    const hadNote = Boolean(notes);
    emptyNote(hadNote ? "대화가 변경되었습니다." : "대화를 기록으로 정리합니다.", "현재 텍스트로 한국어와 영어 SOAP 초안을 생성할 수 있습니다.");
    $("transcript-meta").textContent = "직접 편집";
    clearError();
    setPipeline(transcript.value.trim() ? 2 : 0);
    setStatus(hadNote ? "텍스트가 변경되어 기존 초안을 초기화했습니다. 다시 생성해 주세요." : "대화 내용을 확인한 후 ‘한·영 SOAP 생성’을 눌러 주세요.");
  });
  $("generate").addEventListener("click", async () => {
    if (busy || !transcript.value.trim()) return;
    const operation = newOperation();
    try { await generateNotes(operation); }
    catch (error) {
      if (operation !== sequence || error.name === "AbortError") return;
      emptyNote("초안 생성을 완료하지 못했습니다.", "전사 텍스트는 유지됩니다. 내용을 확인하고 다시 생성해 주세요.");
      setBusy("", "전사 텍스트는 유지됩니다. 확인 후 다시 생성해 주세요.");
      showError(error);
    }
  });
  $("example").addEventListener("click", () => {
    if (busy) return;
    newOperation();
    removeAudio();
    transcript.value = $("input-language").value === "en"
      ? "Doctor: What brings you in today?\nPatient: I have had a sore throat and a dry cough for three days.\nDoctor: Any fever or shortness of breath?\nPatient: No fever or shortness of breath.\nDoctor: Do you take any medications or have any allergies?\nPatient: No regular medications and no known drug allergies.\nDoctor: Your temperature is 36.7 degrees Celsius. Your throat is mildly red, and your lungs are clear. This is likely a viral upper respiratory infection. Please rest, drink fluids, and return if you develop a high fever, shortness of breath, or worsening symptoms."
      : "의사: 오늘 어떤 증상으로 오셨나요?\n환자: 사흘 전부터 목이 아프고 마른기침이 나요.\n의사: 열이 나거나 숨이 차지는 않으셨나요?\n환자: 열은 없었고 숨이 차지도 않았어요.\n의사: 평소 드시는 약이나 약물 알레르기가 있나요?\n환자: 평소 먹는 약은 없고, 알고 있는 약물 알레르기도 없어요.\n의사: 체온은 36.7도이고, 목이 약간 붉지만 폐 청진은 정상입니다. 바이러스성 상기도 감염이 의심됩니다. 충분히 쉬고 수분을 섭취하세요. 고열이나 호흡곤란이 생기거나 증상이 악화되면 다시 내원해 주세요.";
    $("transcript-meta").textContent = "가상 진료 예시";
    emptyNote("가상 진료 예시가 준비되었습니다.", "생성 버튼을 누르면 실제 EMR 모델이 두 언어의 초안을 작성합니다.");
    setPipeline(2);
    setStatus("학습용 가상 대화입니다. ‘한·영 SOAP 생성’을 눌러 실제 모델 결과를 확인하세요.");
    transcript.focus();
  });
  $("reset").addEventListener("click", () => {
    newOperation();
    removeAudio();
    transcript.value = "";
    $("transcript-meta").textContent = "입력 대기";
    $("recording-state").lastChild.textContent = " 녹음 준비";
    const total = document.createElement("span");
    total.textContent = ` / ${formatTime(limits.audio_seconds)}`;
    $("timer").replaceChildren(document.createTextNode("00:00"), total);
    emptyNote();
    selectNote("ko");
    setPipeline(0);
    setStatus("마이크를 켜거나 음성 파일을 올려 시작하세요.");
  });
  for (const language of ["ko", "en"]) {
    $(`tab-${language}`).addEventListener("click", () => selectNote(language));
    $(`tab-${language}`).addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        selectNote(event.key === "Home" ? "ko" : event.key === "End" ? "en" : language === "ko" ? "en" : "ko", true);
      }
    });
  }
  $("copy").addEventListener("click", async () => {
    if (!notes) return;
    try {
      await navigator.clipboard.writeText(notes[noteLanguage]);
      setStatus(`${noteLanguage === "ko" ? "한국어" : "영어"} SOAP 초안을 복사했습니다.`);
    } catch { showError(new Error("클립보드에 접근할 수 없습니다. 초안을 직접 선택해 복사하거나 TXT로 다운로드해 주세요.")); }
  });
  $("download").addEventListener("click", () => {
    if (!notes) return;
    const text = `NLP&AI — SOAP EMR 검토용 초안\n의료진의 검토 후 사용하세요.\n\n한국어\n${"─".repeat(30)}\n${notes.ko}\n\nEnglish\n${"─".repeat(30)}\n${notes.en}\n`;
    const url = URL.createObjectURL(new Blob(["\uFEFF", text], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    const today = new Date();
    const date = [today.getFullYear(), today.getMonth() + 1, today.getDate()].map((value) => String(value).padStart(2, "0")).join("-");
    link.download = `nlp-ai-soap-${date}.txt`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus("한국어와 영어 SOAP 초안을 하나의 TXT 파일로 다운로드했습니다.");
  });
  $("warmup").addEventListener("click", async () => {
    if (busy) return;
    const operation = newOperation();
    setBusy("warmup", "음성·EMR 모델을 준비하고 있습니다. 첫 실행에는 로딩에 수 분이 걸릴 수 있습니다.");
    try {
      const result = await api("/api/warmup", { method: "POST" }, controller.signal);
      if (operation !== sequence) return;
      showHealth(result);
      setBusy("", "모델 준비가 끝났습니다. 녹음을 시작해 주세요.");
    } catch (error) {
      if (operation !== sequence || error.name === "AbortError") return;
      setBusy("", "모델 준비를 완료하지 못했습니다.");
      showError(error);
      refreshHealth();
    }
  });
  window.addEventListener("resize", () => { if (!analyser) drawWaveform(); });
  window.addEventListener("pagehide", () => {
    ++sequence;
    controller?.abort();
    clearInterval(clock);
    if (recorder?.state === "recording") recorder.stop();
    releaseMicrophone();
    removeAudio();
  });
  drawWaveform();
  refreshHealth();
})();
