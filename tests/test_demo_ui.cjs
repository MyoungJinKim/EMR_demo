// Run with node tests/test_demo_ui.cjs; browser state checks without dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor() {
    this.value = "";
    this.textContent = "";
    this.hidden = false;
    this.events = {};
    this.attributes = {};
    this.children = [];
    this.lastChild = { textContent: "" };
    this.classList = { toggle() {} };
    this.clientWidth = 600;
    this.clientHeight = 70;
    this.currentTime = 0;
    this.pauseCalls = 0;
    this.playCalls = 0;
  }
  get textContent() { return this.children?.length ? this.children.map((child) => child.textContent).join("") : this._textContent; }
  set textContent(value) { this._textContent = value; this.children = []; }
  set innerHTML(_) { throw new Error("Untrusted UI text must not be rendered as HTML"); }
  addEventListener(name, callback) { this.events[name] = callback; }
  async dispatch(name, event = {}) { return this.events[name]?.(event); }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  replaceChildren(...children) { this.children = children; }
  appendChild(child) { this.children.push(child); }
  remove() {}
  click() { return this.dispatch("click"); }
  focus() {}
  pause() { this.pauseCalls++; }
  async play() { this.playCalls++; }
  load() {}
  getContext() { return new Proxy({}, { get: () => () => {} }); }
}

const recorders = [];
class FakeMediaRecorder extends Element {
  static isTypeSupported(type) { return type === "audio/webm;codecs=opus"; }
  constructor(stream, options) {
    super();
    this.stream = stream;
    this.mimeType = options.mimeType;
    this.state = "inactive";
    this.stopCalls = 0;
    recorders.push(this);
  }
  start(timeslice) {
    assert.equal(timeslice, 1000);
    this.state = "recording";
  }
  stop() {
    assert.equal(this.state, "recording", "Recorder must stop only while active");
    this.stopCalls++;
    this.state = "inactive";
    queueMicrotask(() => {
      this.events.dataavailable({ data: new Blob(["final chunk"], { type: this.mimeType }) });
      this.events.stop();
    });
  }
}
const audioContexts = [];
class FakeAudioContext {
  constructor() { this.state = "suspended"; this.closeCalls = 0; audioContexts.push(this); }
  async resume() { this.state = "running"; }
  async close() { this.state = "closed"; this.closeCalls++; }
  createAnalyser() { return { frequencyBinCount: 256, getByteTimeDomainData: (values) => values.fill(128) }; }
  createMediaStreamSource() { return { connect() {} }; }
}

const staticDir = path.resolve(__dirname, "../src/mediqa_qwen_eval/demo_static");
const html = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
const elements = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element()]));
elements["input-language"].value = "ko";
const requests = [];
const health = { ready: true, stt: { model: "large-v3", loaded: false }, emr: { model: "Qwen", loaded: true, backend: "transformers" } };
const soapNotes = {
  ko: "S · 주관적 정보\n기침. <img src=x>\n\nO · 객관적 소견\n언급 없음.\n\nA · 평가\n언급 없음.\n\nP · 계획\n휴식 안내.",
  en: "S · Subjective\nCough.\n\nO · Objective\nNot documented.\n\nA · Assessment\nNot documented.\n\nP · Plan\nRest advised."
};
let generateResponse = () => Promise.resolve({ ok: true, json: async () => ({ notes: soapNotes, elapsed_seconds: 1.5, warnings: [] }) });
let transcribeResponse = async () => ({ ok: true, json: async () => ({ text: "음성에서 전사한 대화", duration_seconds: 5 }) });
let copiedText = "";
const downloads = [];
let getMicrophone;
const context = {
  document: { getElementById: (id) => elements[id], body: new Element(), createElement: () => new Element(), createTextNode: (value) => ({ textContent: value }) },
  window: { isSecureContext: true, devicePixelRatio: 1, addEventListener() {}, MediaRecorder: FakeMediaRecorder, AudioContext: FakeAudioContext },
  navigator: { clipboard: { writeText: async (text) => { copiedText = text; } }, mediaDevices: { getUserMedia: () => new Promise((resolve) => { getMicrophone = resolve; }) } },
  fetch: async (url, options = {}) => {
    requests.push({ url, ...options });
    if (url === "/api/health") return { ok: true, json: async () => health };
    if (url.startsWith("/api/transcribe")) return transcribeResponse();
    if (url === "/api/generate") return generateResponse();
    throw new Error(`Unexpected request ${url}`);
  },
  Blob, AbortController, MediaRecorder: FakeMediaRecorder, URL: { createObjectURL: (blob) => { downloads.push(blob); return `blob:test-${downloads.length}`; }, revokeObjectURL() {} },
  performance, console, setInterval: () => 1, clearInterval() {}, setTimeout, clearTimeout,
  requestAnimationFrame: () => 1, cancelAnimationFrame() {}
};
vm.runInNewContext(fs.readFileSync(path.join(staticDir, "app.js"), "utf8"), context);
const flush = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };

(async () => {
  await flush();
  assert.equal(elements["health-label"].textContent, "모델 준비 완료", "Prepared models may be unloaded between inference stages");
  delete health.ready;
  health.stt.loaded = true;
  await elements.example.click();
  assert.equal(elements["transcript-meta"].textContent, "가상 진료 예시");
  assert.equal(requests.filter((request) => request.url === "/api/generate").length, 0, "Example must not pretend to generate");

  await elements.generate.click();
  assert.equal(JSON.parse(requests.find((request) => request.url === "/api/generate").body).language, "both");
  assert.equal(elements.note.textContent, soapNotes.ko, "SOAP styling must preserve exact text and escape HTML");
  assert.equal(elements.note.children.filter((child) => child.className === "soap-heading").length, 4);
  assert.ok(elements.note.children.filter((child) => child.className === "soap-heading").every((child) => child.attributes.role === "heading" && child.attributes["aria-level"] === "3"));
  assert.equal(elements.copy.disabled, false);
  await elements["tab-en"].click();
  assert.equal(elements.note.textContent, soapNotes.en);
  assert.equal(elements.note.children.filter((child) => child.className === "soap-heading").length, 4);
  await elements.copy.click();
  assert.equal(copiedText, soapNotes.en, "Copy must retain raw SOAP text");
  await elements.download.click();
  const downloadedText = await downloads.at(-1).text();
  assert.ok(downloadedText.includes(soapNotes.ko) && downloadedText.includes(soapNotes.en), "Download must contain both complete SOAP notes");
  elements.transcript.value += " 수정";
  await elements.transcript.dispatch("input");
  assert.equal(elements.copy.disabled, true, "Edits must invalidate exportable notes");
  assert.equal(elements.download.disabled, true);
  assert.equal(elements.note.textContent, "");

  const originalText = elements.transcript.value;
  generateResponse = async () => ({ ok: false, status: 503, json: async () => ({ detail: "모델 연결 오류" }) });
  await elements.generate.click();
  assert.equal(elements.transcript.value, originalText, "Generation failure must preserve transcript");
  assert.equal(elements.error.textContent, "모델 연결 오류");
  assert.equal(elements.generate.disabled, false);

  let finishStale;
  generateResponse = () => new Promise((resolve) => { finishStale = resolve; });
  const staleRequest = elements.generate.click();
  await flush();
  await elements.cancel.click();
  generateResponse = async () => ({ ok: true, json: async () => ({ notes: { ko: "새 한국어 초안", en: "New English draft" } }) });
  await elements.generate.click();
  finishStale({ ok: true, json: async () => ({ notes: { ko: "이전 요청", en: "Stale request" } }) });
  await staleRequest;
  assert.equal(elements.note.textContent, "New English draft", "An aborted request must never overwrite a newer result");

  const upload = new Blob(["audio"], { type: "audio/wav" });
  upload.name = "test.wav";
  elements["audio-file"].files = [upload];
  await elements["audio-file"].dispatch("change");
  await flush();
  assert.equal(elements.transcript.value, "음성에서 전사한 대화");
  assert.equal(requests.find((request) => request.url.startsWith("/api/transcribe")).body, upload);
  assert.equal(elements.copy.disabled, false, "Uploading audio must automatically transcribe and generate both notes");
  assert.equal(elements["numeric-review"].hidden, true, "Older STT responses without segments must remain supported");

  const numericResponse = async () => ({ ok: true, json: async () => ({
    text: "와파린 0.5밀리그램. 혈압 120/80입니다.", duration_seconds: 8,
    segments: [
      { start: 0.1, end: 2.2, text: "와파린 0.5밀리그램 <img src=x>" },
      { start: 3, end: 5, text: "혈압 120/80입니다." },
      { start: 6, end: 7, text: "기침은 없습니다." },
      { start: -1, end: 2, text: "잘못된 시간 7" },
      { start: 4, end: 3, text: "잘못된 구간 8" },
      { start: "1", end: 3, text: "잘못된 형식 9" },
      null,
    ]
  }) });
  const uploadAgain = async () => { await elements["audio-file"].dispatch("change"); await flush(); };
  transcribeResponse = numericResponse;
  await uploadAgain();
  assert.equal(elements["numeric-review"].hidden, false);
  assert.equal(elements["numeric-review"].open, false, "Numeric review should start collapsed");
  assert.equal(elements["numeric-segments"].children.length, 2, "Review includes all numeric segments with valid timestamps, including blood pressure");
  const doseButton = elements["numeric-segments"].children[0];
  const bloodPressureButton = elements["numeric-segments"].children[1];
  assert.equal(doseButton.textContent, "00:00 · 와파린 0.5밀리그램 <img src=x>", "Review must preserve recognized numbers and render text safely");
  const player = elements["audio-player"];
  await doseButton.click();
  assert.equal(player.currentTime, 0, "Review seeks 0.3 seconds before speech without seeking below zero");
  await bloodPressureButton.click();
  assert.equal(player.currentTime, 2.7);
  const pausesBeforeEnd = player.pauseCalls;
  player.currentTime = 5.3;
  await player.dispatch("timeupdate");
  assert.equal(player.pauseCalls, pausesBeforeEnd, "Review preserves the 0.4-second trailing context");
  player.currentTime = 5.4;
  await player.dispatch("timeupdate");
  assert.equal(player.pauseCalls, pausesBeforeEnd + 1, "Review pauses at the selected segment end plus context");
  player.currentTime = 7;
  await player.dispatch("timeupdate");
  assert.equal(player.pauseCalls, pausesBeforeEnd + 1, "A completed review must not affect later manual playback");

  await doseButton.click();
  elements.transcript.value += " 수정";
  await elements.transcript.dispatch("input");
  assert.equal(elements["numeric-review"].hidden, true, "Edited text must not retain original-transcript review buttons");
  assert.equal(elements["numeric-segments"].children.length, 0);
  assert.equal(player.pauseCalls, pausesBeforeEnd + 2, "Editing stops a review that no longer matches the text");
  const playsBeforeStale = player.playCalls;
  await doseButton.click();
  assert.equal(player.playCalls, playsBeforeStale, "Detached review buttons cannot replay stale audio");

  await uploadAgain();
  await doseButton.click();
  assert.equal(player.playCalls, playsBeforeStale, "A previous file's button must not seek or play a new audio source");
  const originalPlay = player.play;
  player.play = async () => { throw new Error("Playback unavailable"); };
  await elements["numeric-segments"].children[0].click();
  assert.match(elements["status-text"].textContent, /원음 재생을 시작하지 못했습니다/);
  assert.equal(elements.transcript.value, "와파린 0.5밀리그램. 혈압 120/80입니다.", "Playback failure must preserve the recognized transcript");
  player.play = originalPlay;
  let finishTranscription;
  transcribeResponse = () => new Promise((resolve) => { finishTranscription = resolve; });
  await elements["audio-file"].dispatch("change");
  assert.equal(elements["numeric-review"].hidden, true, "Starting another upload immediately clears previous review segments");
  finishTranscription({ ok: false, status: 400, json: async () => ({ detail: "음성 인식 실패" }) });
  await flush();
  assert.equal(elements["numeric-review"].hidden, true, "Transcription failure must leave no stale review segments");
  assert.equal(elements["numeric-segments"].children.length, 0);

  transcribeResponse = numericResponse;
  await uploadAgain();
  await elements.reset.click();
  assert.equal(elements["numeric-review"].hidden, true, "Reset clears audio review");
  await uploadAgain();
  await elements.example.click();
  assert.equal(elements["numeric-review"].hidden, true, "Text-only examples clear audio review");
  await elements["tab-en"].click();
  transcribeResponse = async () => ({ ok: true, json: async () => ({ text: "음성에서 전사한 대화", duration_seconds: 5 }) });

  let stopped = false;
  const pendingPermission = elements.record.click();
  await flush();
  await elements.cancel.click();
  getMicrophone({ getTracks: () => [{ stop: () => { stopped = true; } }] });
  await pendingPermission;
  assert.equal(stopped, true, "Cancelled microphone permission must release late-arriving tracks");
  assert.equal(elements.record.disabled, false);

  const tracks = [];
  context.navigator.mediaDevices.getUserMedia = async () => {
    const track = { stopCalls: 0, stop() { this.stopCalls++; } };
    tracks.push(track);
    return { getTracks: () => [track] };
  };
  const transcribeCount = () => requests.filter((request) => request.url.startsWith("/api/transcribe")).length;
  const beforeRecording = transcribeCount();
  await elements.record.click();
  assert.equal(recorders[0].state, "recording");
  assert.equal(elements["record-label"].textContent, "녹음 종료 · 생성");
  await recorders[0].dispatch("dataavailable", { data: new Blob(["first chunk "]) });
  await elements.record.click();
  await flush();
  assert.equal(recorders[0].stopCalls, 1);
  assert.equal(tracks[0].stopCalls, 1, "Stopping recording must release the microphone once");
  assert.equal(audioContexts[0].closeCalls, 1, "Stopping recording must close its audio context");
  assert.equal(transcribeCount(), beforeRecording + 1);
  const audioRequest = requests.filter((request) => request.url.startsWith("/api/transcribe")).at(-1);
  assert.equal(audioRequest.headers["Content-Type"], "audio/webm;codecs=opus");
  assert.equal(await audioRequest.body.text(), "first chunk final chunk", "The final recorder chunk must be included");
  assert.equal(elements.transcript.value, "음성에서 전사한 대화");
  assert.equal(elements.note.textContent, "New English draft");
  assert.equal(elements.copy.disabled, false, "Stopping recording must automatically finish both-language generation");

  await elements.record.click();
  await elements.cancel.click();
  await flush();
  assert.equal(recorders[1].stopCalls, 1);
  assert.equal(tracks[1].stopCalls, 1);
  assert.equal(audioContexts[1].closeCalls, 1);
  assert.equal(transcribeCount(), beforeRecording + 1, "Cancelled recording must never upload audio");
  await elements.record.click();
  await recorders[1].dispatch("error");
  assert.equal(recorders[2].state, "recording", "An old recorder error must not cancel a new recording");
  assert.equal(elements["record-label"].textContent, "녹음 종료 · 생성");
  assert.equal(elements.error.hidden, true);
  await elements.cancel.click();
  await flush();
  assert.equal(recorders[2].stopCalls, 1);
  assert.equal(tracks[2].stopCalls, 1);
  assert.equal(audioContexts[2].closeCalls, 1);
  assert.equal(transcribeCount(), beforeRecording + 1);
  console.log("Demo UI checks passed: bilingual SOAP, safe text, copy/download, numeric-segment replay/stop/invalidation, health, errors, upload, recording lifecycle, resource cleanup, and cancellation.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
