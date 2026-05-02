(async () => {
  "use strict";

  const RATE = 16000;
  const CHUNK = 512;
  const VAD_THRESHOLD = 0.5;
  const PRE_SPEECH_MS = 200;
  const STOP_MS = 1000;
  const MAX_DURATION_SECONDS = 8;
  const MODEL_RESET_STATES_TIME = 5000;
  const SMART_INPUT_SAMPLES = RATE * MAX_DURATION_SECONDS;
  const SMART_INPUT_FRAMES = SMART_INPUT_SAMPLES / 160;
  const ORT_WEBGPU_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260410-5e55544225/dist/ort.webgpu.min.mjs";
  const ORT_WEBGPU_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260410-5e55544225/dist/";
  const APP_VERSION = document.querySelector('meta[name="app-version"]')?.content || "dev";

  const MODEL_PATHS = {
    vad: withAssetVersion("./models/silero_vad.onnx"),
    smartTurnCpu: withAssetVersion("./models/smart-turn-v3.2-cpu.onnx"),
    smartTurnGpu: withAssetVersion("./models/smart-turn-v3.2-gpu.onnx"),
  };

  let ort = window.ort;

  const els = {
    startButton: document.getElementById("startButton"),
    stopButton: document.getElementById("stopButton"),
    stateText: document.getElementById("stateText"),
    vadText: document.getElementById("vadText"),
    predictionText: document.getElementById("predictionText"),
    probabilityText: document.getElementById("probabilityText"),
    vadAvgText: document.getElementById("vadAvgText"),
    smartTurnAvgText: document.getElementById("smartTurnAvgText"),
    featureAvgText: document.getElementById("featureAvgText"),
    backendText: document.getElementById("backendText"),
    detailText: document.getElementById("detailText"),
    graphCanvas: document.getElementById("graphCanvas"),
    historyList: document.getElementById("historyList"),
    resultMetric: document.querySelector(".result-metric"),
  };

  const state = {
    modelsReady: false,
    running: false,
    loading: false,
    audioContext: null,
    mediaStream: null,
    source: null,
    processor: null,
    resampler: null,
    vad: null,
    smartTurn: null,
    featureExtractor: null,
    inferenceBackend: "--",
    featureBackend: "--",
    vadQueue: [],
    vadBusy: false,
    speechActive: false,
    preBuffer: [],
    segment: [],
    trailingSilence: 0,
    sinceTriggerChunks: 0,
    lastVad: 0,
    lastPrediction: null,
    lastProbability: null,
    inferenceBusy: false,
    uiState: "Idle",
    waveformHistory: [],
    vadHistory: [],
    turnHistory: [],
    timings: {
      vad: { total: 0, count: 0 },
      smartTurn: { total: 0, count: 0 },
      feature: { total: 0, count: 0 },
    },
  };

  const timingOutputs = {
    vad: "vadAvgText",
    smartTurn: "smartTurnAvgText",
    feature: "featureAvgText",
  };

  const chunkMs = (CHUNK / RATE) * 1000;
  const preChunks = Math.ceil(PRE_SPEECH_MS / chunkMs);
  const stopChunks = Math.ceil(STOP_MS / chunkMs);
  const maxChunks = Math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE));

  checkForPageUpdate();

  function withAssetVersion(path) {
    if (APP_VERSION === "dev") {
      return path;
    }
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}v=${encodeURIComponent(APP_VERSION)}`;
  }

  async function checkForPageUpdate() {
    if (APP_VERSION === "dev" || location.protocol === "file:") {
      return;
    }

    try {
      const response = await fetch(withCacheBust("./index.html"), { cache: "no-store" });
      if (!response.ok) {
        return;
      }
      const html = await response.text();
      const latestVersion = parseAppVersion(html);
      if (latestVersion && latestVersion !== APP_VERSION) {
        const url = new URL(location.href);
        url.searchParams.set("v", latestVersion);
        location.replace(url.toString());
      }
    } catch (error) {
      console.warn("Could not check for page update.", error);
    }
  }

  function withCacheBust(path) {
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}t=${Date.now()}`;
  }

  function parseAppVersion(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");
    return doc.querySelector('meta[name="app-version"]')?.content || "";
  }

  class SileroVAD {
    constructor(session) {
      this.session = session;
      this.contextSize = 64;
      this.lastReset = performance.now();
      this.reset();
    }

    reset() {
      this.vadState = new Float32Array(2 * 1 * 128);
      this.context = new Float32Array(this.contextSize);
      this.lastReset = performance.now();
    }

    maybeReset() {
      if (performance.now() - this.lastReset >= MODEL_RESET_STATES_TIME) {
        this.reset();
      }
    }

    async probability(chunk) {
      if (chunk.length !== CHUNK) {
        throw new Error(`Expected ${CHUNK} samples, got ${chunk.length}`);
      }

      const input = new Float32Array(this.contextSize + CHUNK);
      input.set(this.context, 0);
      input.set(chunk, this.contextSize);

      const feeds = {
        input: new ort.Tensor("float32", input, [1, this.contextSize + CHUNK]),
        state: new ort.Tensor("float32", this.vadState, [2, 1, 128]),
        sr: new ort.Tensor("int64", BigInt64Array.from([16000n]), []),
      };

      const outputs = await this.session.run(feeds);
      const names = this.session.outputNames;
      const probTensor = outputs[names[0]];
      const stateTensor = outputs[names[1]];

      this.vadState = new Float32Array(stateTensor.data);
      this.context = input.slice(CHUNK);
      this.maybeReset();

      return Number(probTensor.data[0]);
    }
  }

  class StreamingResampler {
    constructor(sourceRate, targetRate) {
      this.sourceRate = sourceRate;
      this.targetRate = targetRate;
      this.step = sourceRate / targetRate;
      this.pending = new Float32Array(0);
      this.position = 0;
    }

    push(input) {
      const merged = new Float32Array(this.pending.length + input.length);
      merged.set(this.pending, 0);
      merged.set(input, this.pending.length);
      this.pending = merged;

      const output = [];
      while (this.position + 1 < this.pending.length) {
        const index = Math.floor(this.position);
        const frac = this.position - index;
        const a = this.pending[index];
        const b = this.pending[index + 1];
        output.push(a + (b - a) * frac);
        this.position += this.step;
      }

      const consumed = Math.floor(this.position);
      if (consumed > 0) {
        this.pending = this.pending.slice(consumed);
        this.position -= consumed;
      }

      return Float32Array.from(output);
    }
  }

  class SmartTurnPredictor {
    constructor(session, extractor) {
      this.session = session;
      this.extractor = extractor;
    }

    async predict(audio) {
      const featureStarted = performance.now();
      const features = await this.extractor.extract(audio);
      const featureMs = performance.now() - featureStarted;
      const input = new ort.Tensor("float32", features, [1, 80, 800]);

      const onnxStarted = performance.now();
      const outputs = await this.session.run({ input_features: input });
      const onnxMs = performance.now() - onnxStarted;
      const firstOutput = outputs[this.session.outputNames[0]];
      const probability = Number(firstOutput.data[0]);
      return {
        prediction: probability > 0.5 ? 1 : 0,
        probability,
        timings: {
          featureMs,
          onnxMs,
        },
      };
    }
  }

  class BuiltInWhisperFeatureExtractor {
    constructor() {
      this.nFft = 400;
      this.hopLength = 160;
      this.nMels = 80;
      this.nFrames = 800;
      this.eps = 1e-10;
      this.window = makeHannWindow(this.nFft);
      this.fftPlan = makeFft400Plan();
      this.melFilters = makeMelFilters(RATE, this.nFft, this.nMels);
    }

    extract(audio) {
      const samples = normalizePaddedAudio(audio, SMART_INPUT_SAMPLES);
      const padded = reflectPad(samples, this.nFft / 2);
      const frameCount = Math.floor((padded.length - this.nFft) / this.hopLength) + 1;
      const frameLimit = Math.min(frameCount - 1, this.nFrames);
      const mel = new Float32Array(this.nMels * this.nFrames);
      const power = new Float32Array(this.nFft / 2 + 1);
      let globalMax = -Infinity;

      for (let frame = 0; frame < frameLimit; frame += 1) {
        const start = frame * this.hopLength;
        powerSpectrumInto(power, padded, start, this.window, this.fftPlan);
        for (let m = 0; m < this.nMels; m += 1) {
          let energy = 0;
          const filter = this.melFilters[m];
          const weights = filter.weights;
          const filterStart = filter.start;
          for (let i = 0; i < weights.length; i += 1) {
            energy += weights[i] * power[filterStart + i];
          }
          const value = Math.log10(Math.max(energy, this.eps));
          mel[m * this.nFrames + frame] = value;
          if (value > globalMax) {
            globalMax = value;
          }
        }
      }

      const floor = globalMax - 8;
      for (let i = 0; i < mel.length; i += 1) {
        mel[i] = (Math.max(mel[i], floor) + 4) / 4;
      }
      return mel;
    }
  }

  function normalizePaddedAudio(audio, length) {
    const out = new Float32Array(length);
    let sum = 0;
    if (audio.length >= length) {
      const start = audio.length - length;
      for (let i = 0; i < length; i += 1) {
        const value = audio[start + i];
        out[i] = value;
        sum += value;
      }
    } else {
      const offset = length - audio.length;
      for (let i = 0; i < audio.length; i += 1) {
        const value = audio[i];
        out[offset + i] = value;
        sum += value;
      }
    }

    const mean = sum / length;
    let variance = 0;
    for (let i = 0; i < out.length; i += 1) {
      const diff = out[i] - mean;
      variance += diff * diff;
    }
    const scale = Math.sqrt(variance / length + 1e-7);
    for (let i = 0; i < out.length; i += 1) {
      out[i] = (out[i] - mean) / scale;
    }
    return out;
  }

  function reflectPad(audio, pad) {
    const out = new Float32Array(audio.length + pad * 2);
    for (let i = 0; i < pad; i += 1) {
      out[i] = audio[pad - i];
      out[pad + audio.length + i] = audio[audio.length - 2 - i];
    }
    out.set(audio, pad);
    return out;
  }

  function makeHannWindow(size) {
    const win = new Float32Array(size);
    for (let i = 0; i < size; i += 1) {
      win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
    }
    return win;
  }

  // 400 = 16 * 25, matching Whisper's n_fft without zero-padding to a different model input.
  function makeFft400Plan() {
    const n1 = 16;
    const n2 = 25;
    const bins = 201;
    const table1 = makeDftTable(n1);
    const secondStage = makeSecondStageTable(n1, n2, bins);
    return {
      n1,
      n2,
      bins,
      cos1: table1.cos,
      sin1: table1.sin,
      secondStageCos: secondStage.cos,
      secondStageSin: secondStage.sin,
      stageReal: new Float32Array(n1 * n2),
      stageImag: new Float32Array(n1 * n2),
      frameValues: new Float32Array(n1),
    };
  }

  function makeDftTable(size) {
    const cos = new Float32Array(size * size);
    const sin = new Float32Array(size * size);
    for (let k = 0; k < size; k += 1) {
      const row = k * size;
      for (let n = 0; n < size; n += 1) {
        const phase = (2 * Math.PI * k * n) / size;
        cos[row + n] = Math.cos(phase);
        sin[row + n] = Math.sin(phase);
      }
    }
    return { cos, sin };
  }

  function makeSecondStageTable(n1, n2, bins) {
    const cos = new Float32Array(bins * n2);
    const sin = new Float32Array(bins * n2);
    const n = n1 * n2;
    for (let bin = 0; bin < bins; bin += 1) {
      const k1 = bin % n1;
      const k2 = Math.floor(bin / n1);
      const row = bin * n2;
      for (let n2Index = 0; n2Index < n2; n2Index += 1) {
        const phase = 2 * Math.PI * ((n2Index * k1) / n + (n2Index * k2) / n2);
        cos[row + n2Index] = Math.cos(phase);
        sin[row + n2Index] = Math.sin(phase);
      }
    }
    return { cos, sin };
  }

  function powerSpectrumInto(power, samples, start, window, plan) {
    const { n1, n2, bins, cos1, sin1, secondStageCos, secondStageSin, stageReal, stageImag, frameValues } = plan;

    for (let n2Index = 0; n2Index < n2; n2Index += 1) {
      for (let n1Index = 0; n1Index < n1; n1Index += 1) {
        const sampleIndex = n2Index + n2 * n1Index;
        frameValues[n1Index] = samples[start + sampleIndex] * window[sampleIndex];
      }
      for (let k1 = 0; k1 < n1; k1 += 1) {
        let real = 0;
        let imag = 0;
        const tableOffset = k1 * n1;
        for (let n1Index = 0; n1Index < n1; n1Index += 1) {
          const tableIndex = tableOffset + n1Index;
          const value = frameValues[n1Index];
          real += value * cos1[tableIndex];
          imag -= value * sin1[tableIndex];
        }
        const stageIndex = k1 * n2 + n2Index;
        stageReal[stageIndex] = real;
        stageImag[stageIndex] = imag;
      }
    }

    for (let bin = 0; bin < bins; bin += 1) {
      const k1 = bin % n1;
      const stageOffset = k1 * n2;
      const tableOffset = bin * n2;
      let real = 0;
      let imag = 0;
      for (let n2Index = 0; n2Index < n2; n2Index += 1) {
        const inputIndex = stageOffset + n2Index;
        const tableIndex = tableOffset + n2Index;
        const inputReal = stageReal[inputIndex];
        const inputImag = stageImag[inputIndex];
        const tableReal = secondStageCos[tableIndex];
        const tableImag = secondStageSin[tableIndex];
        real += inputReal * tableReal + inputImag * tableImag;
        imag += inputImag * tableReal - inputReal * tableImag;
      }
      power[bin] = real * real + imag * imag;
    }
  }

  function makeMelFilters(sampleRate, nFft, nMels) {
    const bins = nFft / 2 + 1;
    const fftFreqs = new Float32Array(bins);
    for (let i = 0; i < bins; i += 1) {
      fftFreqs[i] = (sampleRate / nFft) * i;
    }

    const melMin = hzToMel(0);
    const melMax = hzToMel(sampleRate / 2);
    const melPoints = new Float32Array(nMels + 2);
    const hzPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < melPoints.length; i += 1) {
      melPoints[i] = melMin + ((melMax - melMin) * i) / (nMels + 1);
      hzPoints[i] = melToHz(melPoints[i]);
    }

    const filters = [];
    for (let m = 0; m < nMels; m += 1) {
      const left = hzPoints[m];
      const center = hzPoints[m + 1];
      const right = hzPoints[m + 2];
      const enorm = 2 / (right - left);
      const filter = new Float32Array(bins);
      for (let k = 0; k < bins; k += 1) {
        const freq = fftFreqs[k];
        const lower = (freq - left) / (center - left);
        const upper = (right - freq) / (right - center);
        filter[k] = Math.max(0, Math.min(lower, upper)) * enorm;
      }

      let start = 0;
      while (start < filter.length && filter[start] === 0) {
        start += 1;
      }
      let end = filter.length - 1;
      while (end >= start && filter[end] === 0) {
        end -= 1;
      }
      filters.push({
        start,
        weights: start <= end ? filter.slice(start, end + 1) : new Float32Array(0),
      });
    }
    return filters;
  }

  function hzToMel(hz) {
    const fSp = 200 / 3;
    const minLogHz = 1000;
    const minLogMel = minLogHz / fSp;
    const logStep = Math.log(6.4) / 27;
    if (hz < minLogHz) {
      return hz / fSp;
    }
    return minLogMel + Math.log(hz / minLogHz) / logStep;
  }

  function melToHz(mel) {
    const fSp = 200 / 3;
    const minLogHz = 1000;
    const minLogMel = minLogHz / fSp;
    const logStep = Math.log(6.4) / 27;
    if (mel < minLogMel) {
      return mel * fSp;
    }
    return minLogHz * Math.exp(logStep * (mel - minLogMel));
  }

  async function ensureFeatureExtractor() {
    if (state.featureExtractor) {
      return;
    }

    state.featureExtractor = new BuiltInWhisperFeatureExtractor();
    state.featureBackend = "Built-in";
    updateBackendText();
  }

  async function ensureOrtRuntime() {
    if (state.vad && state.smartTurn) {
      return;
    }

    const wantsWebGpu = await canUseWebGpu();
    if (wantsWebGpu) {
      try {
        const webgpuOrt = await import(ORT_WEBGPU_URL);
        configureOrt(webgpuOrt, "webgpu");
        await createModelSessions(webgpuOrt, "webgpu");
        return;
      } catch (error) {
        console.warn("WebGPU backend failed; falling back to WASM.", error);
      }
    }

    const wasmOrt = window.ort || ort;
    if (!wasmOrt) {
      throw new Error("ONNX Runtime Web is not available.");
    }
    configureOrt(wasmOrt, "wasm");
    await createModelSessions(wasmOrt, "wasm");
  }

  async function canUseWebGpu() {
    if (!navigator.gpu) {
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      return Boolean(adapter);
    } catch (error) {
      console.warn("WebGPU adapter check failed.", error);
      return false;
    }
  }

  function configureOrt(ortRuntime, backend) {
    ort = ortRuntime;
    if (backend === "webgpu") {
      ort.env.wasm.wasmPaths = {
        mjs: `${ORT_WEBGPU_WASM_BASE}ort-wasm-simd-threaded.jsep.mjs`,
        wasm: `${ORT_WEBGPU_WASM_BASE}ort-wasm-simd-threaded.jsep.wasm`,
      };
      ort.env.webgpu.powerPreference = "high-performance";
    } else {
      ort.env.wasm.wasmPaths = "./vendor/";
    }
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
  }

  async function createModelSessions(ortRuntime, backend) {
    const sessionOptions = {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
    };
    const smartTurnModelPath = backend === "webgpu" ? MODEL_PATHS.smartTurnGpu : MODEL_PATHS.smartTurnCpu;
    const vadSession = await ortRuntime.InferenceSession.create(MODEL_PATHS.vad, sessionOptions);
    const smartSession = await ortRuntime.InferenceSession.create(smartTurnModelPath, sessionOptions);
    if (backend === "webgpu") {
      await validateModelSessions(ortRuntime, vadSession, smartSession);
    }
    state.vad = new SileroVAD(vadSession);
    state.smartTurn = new SmartTurnPredictor(smartSession, state.featureExtractor);
    state.inferenceBackend = backend === "webgpu" ? "WebGPU" : "Wasm";
    updateBackendText();
  }

  async function validateModelSessions(ortRuntime, vadSession, smartSession) {
    await vadSession.run({
      input: new ortRuntime.Tensor("float32", new Float32Array(64 + CHUNK), [1, 64 + CHUNK]),
      state: new ortRuntime.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]),
      sr: new ortRuntime.Tensor("int64", BigInt64Array.from([16000n]), []),
    });
    await smartSession.run({
      input_features: new ortRuntime.Tensor("float32", new Float32Array(80 * SMART_INPUT_FRAMES), [1, 80, SMART_INPUT_FRAMES]),
    });
  }

  function updateBackendText() {
    els.backendText.textContent = state.inferenceBackend;
  }

  async function ensureModels() {
    if (state.modelsReady || state.loading) {
      return;
    }

    state.loading = true;
    setUiState("Loading models");
    try {
      await ensureFeatureExtractor();
      await ensureOrtRuntime();
      state.modelsReady = true;
      setUiState("Ready");
    } finally {
      state.loading = false;
    }
  }

  async function start() {
    try {
      els.startButton.disabled = true;
      await ensureModels();

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass || !navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not expose microphone capture APIs.");
      }

      state.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      state.audioContext = new AudioContextClass();
      if (state.audioContext.state === "suspended") {
        await state.audioContext.resume();
      }

      state.resampler = new StreamingResampler(state.audioContext.sampleRate, RATE);
      state.source = state.audioContext.createMediaStreamSource(state.mediaStream);
      state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
      state.processor.onaudioprocess = onAudioProcess;
      state.source.connect(state.processor);
      state.processor.connect(state.audioContext.destination);

      resetCaptureState();
      resetRuntimeStats();
      state.running = true;
      els.stopButton.disabled = false;
      setUiState("Listening");
      els.detailText.textContent = `${state.uiState} | Input ${Math.round(state.audioContext.sampleRate)} Hz -> ${RATE} Hz | Backend ${state.inferenceBackend} | Features ${state.featureBackend}`;
    } catch (error) {
      stop();
      setUiState("Error");
      els.detailText.textContent = error.message || String(error);
      els.startButton.disabled = false;
    }
  }

  function stop() {
    state.running = false;
    if (state.processor) {
      state.processor.disconnect();
      state.processor.onaudioprocess = null;
      state.processor = null;
    }
    if (state.source) {
      state.source.disconnect();
      state.source = null;
    }
    if (state.mediaStream) {
      for (const track of state.mediaStream.getTracks()) {
        track.stop();
      }
      state.mediaStream = null;
    }
    if (state.audioContext) {
      state.audioContext.close();
      state.audioContext = null;
    }
    state.resampler = null;
    state.vadQueue = [];
    state.vadBusy = false;
    resetCaptureState();
    els.startButton.disabled = false;
    els.stopButton.disabled = true;
    setUiState(state.modelsReady ? "Ready" : "Idle");
  }

  function resetCaptureState() {
    state.speechActive = false;
    state.preBuffer = [];
    state.segment = [];
    state.trailingSilence = 0;
    state.sinceTriggerChunks = 0;
    if (state.vad) {
      state.vad.reset();
    }
  }

  function onAudioProcess(event) {
    if (!state.running || !state.resampler) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const resampled = state.resampler.push(input);
    appendSamplesToQueue(resampled);

    const output = event.outputBuffer.getChannelData(0);
    output.fill(0);
  }

  function appendSamplesToQueue(samples) {
    for (let i = 0; i < samples.length; i += 1) {
      state.vadQueue.push(samples[i]);
    }
    if (!state.vadBusy) {
      drainVadQueue();
    }
  }

  async function drainVadQueue() {
    state.vadBusy = true;
    try {
      while (state.running && state.vadQueue.length >= CHUNK) {
        const chunk = Float32Array.from(state.vadQueue.slice(0, CHUNK));
        state.vadQueue.splice(0, CHUNK);
        await processChunk(chunk);
      }
    } catch (error) {
      setUiState("Error");
      els.detailText.textContent = error.message || String(error);
      stop();
    } finally {
      state.vadBusy = false;
    }
  }

  async function processChunk(chunk) {
    const rms = Math.sqrt(chunk.reduce((sum, v) => sum + v * v, 0) / chunk.length);
    pushLimited(state.waveformHistory, Math.min(1, rms * 16), 420);

    const vadStarted = performance.now();
    const vadProb = await state.vad.probability(chunk);
    const vadElapsed = performance.now() - vadStarted;
    recordRuntime("vad", vadElapsed);
    state.lastVad = vadProb;
    if (els.vadText) {
      els.vadText.textContent = vadProb.toFixed(2);
    }
    pushLimited(state.vadHistory, vadProb, 420);

    const isSpeech = vadProb > VAD_THRESHOLD;
    if (!state.speechActive) {
      state.preBuffer.push(chunk);
      if (state.preBuffer.length > preChunks) {
        state.preBuffer.shift();
      }
      if (isSpeech) {
        state.segment = state.preBuffer.slice();
        state.segment.push(chunk);
        state.speechActive = true;
        state.trailingSilence = 0;
        state.sinceTriggerChunks = 1;
        setUiState("Recording speech");
      }
      return;
    }

    state.segment.push(chunk);
    state.sinceTriggerChunks += 1;
    if (isSpeech) {
      state.trailingSilence = 0;
    } else {
      state.trailingSilence += 1;
    }

    if (state.trailingSilence >= stopChunks || state.sinceTriggerChunks >= maxChunks) {
      const segment = concatenateChunks(state.segment);
      resetCaptureState();
      await processSegment(segment);
      if (state.running) {
        setUiState("Listening");
      }
    }
  }

  async function processSegment(audio) {
    if (!audio.length || state.inferenceBusy) {
      return;
    }
    state.inferenceBusy = true;
    setUiState("Predicting");
    const started = performance.now();
    try {
      const result = await state.smartTurn.predict(audio);
      const elapsed = performance.now() - started;
      recordRuntime("smartTurn", result.timings.onnxMs);
      recordRuntime("feature", result.timings.featureMs);
      state.lastPrediction = result.prediction;
      state.lastProbability = result.probability;
      pushLimited(state.turnHistory, result.probability, 80);
      renderPrediction(result, audio.length / RATE, elapsed);
    } finally {
      state.inferenceBusy = false;
    }
  }

  function concatenateChunks(chunks) {
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  function renderPrediction(result, durationSec, elapsedMs) {
    const complete = result.prediction === 1;
    if (els.predictionText) {
      els.predictionText.textContent = complete ? "Complete" : "Incomplete";
    }
    if (els.probabilityText) {
      els.probabilityText.textContent = result.probability.toFixed(3);
    }
    if (els.resultMetric) {
      els.resultMetric.classList.toggle("complete", complete);
      els.resultMetric.classList.toggle("incomplete", !complete);
    }
    els.detailText.textContent = `${complete ? "Complete" : "Incomplete"} | p=${result.probability.toFixed(3)} | Segment ${durationSec.toFixed(2)} s | Total ${elapsedMs.toFixed(1)} ms | Feature ${result.timings.featureMs.toFixed(1)} ms | Smart Turn ONNX ${result.timings.onnxMs.toFixed(1)} ms`;

    if (els.historyList) {
      const li = document.createElement("li");
      const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      li.innerHTML = `<span>${time}</span><strong>${complete ? "Complete" : "Incomplete"}</strong><span>${result.probability.toFixed(3)}</span><span>${durationSec.toFixed(2)} s</span>`;
      els.historyList.prepend(li);
      while (els.historyList.children.length > 20) {
        els.historyList.lastElementChild.remove();
      }
    }
  }

  function pushLimited(list, value, maxLength) {
    list.push(value);
    if (list.length > maxLength) {
      list.splice(0, list.length - maxLength);
    }
  }

  function resetRuntimeStats() {
    for (const [name, elementName] of Object.entries(timingOutputs)) {
      state.timings[name].total = 0;
      state.timings[name].count = 0;
      els[elementName].textContent = "--";
    }
  }

  function recordRuntime(name, elapsedMs) {
    const timing = state.timings[name];
    timing.total += elapsedMs;
    timing.count += 1;
    const averageMs = timing.total / timing.count;
    els[timingOutputs[name]].textContent = `${averageMs.toFixed(1)} ms`;
  }

  function setUiState(label) {
    state.uiState = label;
    if (els.stateText) {
      els.stateText.textContent = label;
    }
  }

  function drawGraph() {
    const canvas = els.graphCanvas;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfcfa";
    ctx.fillRect(0, 0, width, height);

    const pad = 36 * dpr;
    const gap = 18 * dpr;
    const laneH = (height - pad * 2 - gap * 2) / 3;
    drawLane(ctx, pad, pad, width - pad * 2, laneH, "Wave RMS", state.waveformHistory, "#237f8f", 0.5);
    drawLane(ctx, pad, pad + laneH + gap, width - pad * 2, laneH, "VAD probability", state.vadHistory, "#cf6d31", VAD_THRESHOLD);
    drawLane(ctx, pad, pad + (laneH + gap) * 2, width - pad * 2, laneH, "Complete probability", state.turnHistory, "#4f6f3d", 0.5, true);

    requestAnimationFrame(drawGraph);
  }

  function drawLane(ctx, x, y, w, h, label, values, color, threshold, bars = false) {
    ctx.save();
    ctx.strokeStyle = "#dfe5df";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#69706d";
    ctx.font = `${12 * (window.devicePixelRatio || 1)}px system-ui, sans-serif`;
    ctx.fillText(label, x + 10, y + 18);

    const thresholdY = y + h - h * threshold;
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(32,36,35,0.3)";
    ctx.beginPath();
    ctx.moveTo(x, thresholdY);
    ctx.lineTo(x + w, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    if (values.length > 0) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      if (bars) {
        const step = w / Math.max(values.length, 12);
        values.forEach((value, index) => {
          const barH = Math.max(2, h * clamp01(value));
          const bx = x + index * step;
          ctx.fillRect(bx + 2, y + h - barH, Math.max(3, step - 4), barH);
        });
      } else {
        ctx.beginPath();
        values.forEach((value, index) => {
          const px = x + (index / Math.max(1, values.length - 1)) * w;
          const py = y + h - h * clamp01(value);
          if (index === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  els.startButton.addEventListener("click", start);
  els.stopButton.addEventListener("click", stop);
  requestAnimationFrame(drawGraph);
})();
