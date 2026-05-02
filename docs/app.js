(() => {
  "use strict";

  const RATE = 16000;
  const CHUNK = 512;
  const VAD_THRESHOLD = 0.5;
  const PRE_SPEECH_MS = 200;
  const STOP_MS = 1000;
  const MAX_DURATION_SECONDS = 8;
  const MODEL_RESET_STATES_TIME = 5000;
  const SMART_INPUT_SAMPLES = RATE * MAX_DURATION_SECONDS;

  const MODEL_PATHS = {
    vad: "./models/silero_vad.onnx",
    smartTurn: "./models/smart-turn-v3.2-cpu.onnx",
  };

  const els = {
    startButton: document.getElementById("startButton"),
    stopButton: document.getElementById("stopButton"),
    stateText: document.getElementById("stateText"),
    vadText: document.getElementById("vadText"),
    predictionText: document.getElementById("predictionText"),
    probabilityText: document.getElementById("probabilityText"),
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
    waveformHistory: [],
    vadHistory: [],
    turnHistory: [],
  };

  const chunkMs = (CHUNK / RATE) * 1000;
  const preChunks = Math.ceil(PRE_SPEECH_MS / chunkMs);
  const stopChunks = Math.ceil(STOP_MS / chunkMs);
  const maxChunks = Math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE));

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
    constructor(session) {
      this.session = session;
      this.extractor = new WhisperFeatureExtractor();
    }

    async predict(audio) {
      const features = await this.extractor.extract(audio);
      const input = new ort.Tensor("float32", features, [1, 80, 800]);
      const outputs = await this.session.run({ input_features: input });
      const firstOutput = outputs[this.session.outputNames[0]];
      const probability = Number(firstOutput.data[0]);
      return {
        prediction: probability > 0.5 ? 1 : 0,
        probability,
      };
    }
  }

  class WhisperFeatureExtractor {
    constructor() {
      this.nFft = 400;
      this.hopLength = 160;
      this.nMels = 80;
      this.nFrames = 800;
      this.eps = 1e-10;
      this.window = makeHannWindow(this.nFft);
      this.dft = makeDftTables(this.nFft);
      this.melFilters = makeMelFilters(RATE, this.nFft, this.nMels);
    }

    async extract(audio) {
      const samples = normalizeAudio(padOrTrimLast(audio, SMART_INPUT_SAMPLES));
      const padded = reflectPad(samples, this.nFft / 2);
      const frameCount = Math.floor((padded.length - this.nFft) / this.hopLength) + 1;
      const mel = new Float32Array(this.nMels * this.nFrames);
      const power = new Float32Array(this.nFft / 2 + 1);
      let globalMax = -Infinity;

      for (let frame = 0; frame < Math.min(frameCount - 1, this.nFrames); frame += 1) {
        const start = frame * this.hopLength;
        powerSpectrumInto(power, padded, start, this.window, this.dft);
        for (let m = 0; m < this.nMels; m += 1) {
          let energy = 0;
          const filter = this.melFilters[m];
          for (let k = 0; k < filter.length; k += 1) {
            energy += filter[k] * power[k];
          }
          const value = Math.log10(Math.max(energy, this.eps));
          mel[m * this.nFrames + frame] = value;
          if (value > globalMax) {
            globalMax = value;
          }
        }
        if (frame > 0 && frame % 40 === 0) {
          await nextAnimationFrame();
        }
      }

      const floor = globalMax - 8;
      for (let i = 0; i < mel.length; i += 1) {
        mel[i] = (Math.max(mel[i], floor) + 4) / 4;
      }
      return mel;
    }
  }

  function padOrTrimLast(audio, length) {
    const out = new Float32Array(length);
    if (audio.length >= length) {
      out.set(audio.slice(audio.length - length));
    } else {
      out.set(audio, length - audio.length);
    }
    return out;
  }

  function normalizeAudio(audio) {
    let sum = 0;
    for (let i = 0; i < audio.length; i += 1) {
      sum += audio[i];
    }
    const mean = sum / audio.length;
    let variance = 0;
    for (let i = 0; i < audio.length; i += 1) {
      const diff = audio[i] - mean;
      variance += diff * diff;
    }
    const scale = Math.sqrt(variance / audio.length + 1e-7);
    const out = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i += 1) {
      out[i] = (audio[i] - mean) / scale;
    }
    return out;
  }

  function reflectPad(audio, pad) {
    const out = new Float32Array(audio.length + pad * 2);
    for (let i = 0; i < pad; i += 1) {
      out[i] = audio[pad - i];
      out[out.length - 1 - i] = audio[audio.length - 2 - i];
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

  function makeDftTables(size) {
    const bins = size / 2 + 1;
    const cos = new Array(bins);
    const sin = new Array(bins);
    for (let k = 0; k < bins; k += 1) {
      cos[k] = new Float32Array(size);
      sin[k] = new Float32Array(size);
      for (let n = 0; n < size; n += 1) {
        const phase = (2 * Math.PI * k * n) / size;
        cos[k][n] = Math.cos(phase);
        sin[k][n] = Math.sin(phase);
      }
    }
    return { cos, sin };
  }

  function powerSpectrumInto(power, samples, start, window, dft) {
    const bins = dft.cos.length;
    for (let k = 0; k < bins; k += 1) {
      let real = 0;
      let imag = 0;
      const cos = dft.cos[k];
      const sin = dft.sin[k];
      for (let n = 0; n < window.length; n += 1) {
        const value = samples[start + n] * window[n];
        real += value * cos[n];
        imag -= value * sin[n];
      }
      power[k] = real * real + imag * imag;
    }
  }

  function nextAnimationFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
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
      filters.push(filter);
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

  async function ensureModels() {
    if (state.modelsReady || state.loading) {
      return;
    }

    state.loading = true;
    setUiState("Loading models");

    ort.env.wasm.wasmPaths = "./vendor/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };
    const vadSession = await ort.InferenceSession.create(MODEL_PATHS.vad, sessionOptions);
    const smartSession = await ort.InferenceSession.create(MODEL_PATHS.smartTurn, sessionOptions);
    state.vad = new SileroVAD(vadSession);
    state.smartTurn = new SmartTurnPredictor(smartSession);
    state.modelsReady = true;
    state.loading = false;
    setUiState("Ready");
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
      state.running = true;
      els.stopButton.disabled = false;
      setUiState("Listening");
      els.detailText.textContent = `Input ${Math.round(state.audioContext.sampleRate)} Hz -> ${RATE} Hz, local WASM CPU inference`;
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

    const vadProb = await state.vad.probability(chunk);
    state.lastVad = vadProb;
    els.vadText.textContent = vadProb.toFixed(2);
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
    els.predictionText.textContent = complete ? "Complete" : "Incomplete";
    els.probabilityText.textContent = result.probability.toFixed(3);
    els.resultMetric.classList.toggle("complete", complete);
    els.resultMetric.classList.toggle("incomplete", !complete);
    els.detailText.textContent = `Segment ${durationSec.toFixed(2)} s, inference ${elapsedMs.toFixed(1)} ms`;

    const li = document.createElement("li");
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    li.innerHTML = `<span>${time}</span><strong>${complete ? "Complete" : "Incomplete"}</strong><span>${result.probability.toFixed(3)}</span><span>${durationSec.toFixed(2)} s</span>`;
    els.historyList.prepend(li);
    while (els.historyList.children.length > 20) {
      els.historyList.lastElementChild.remove();
    }
  }

  function pushLimited(list, value, maxLength) {
    list.push(value);
    if (list.length > maxLength) {
      list.splice(0, list.length - maxLength);
    }
  }

  function setUiState(label) {
    els.stateText.textContent = label;
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
