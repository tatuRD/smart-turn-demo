# Smart Turn Browser Demo

This directory is ready to publish with GitHub Pages.

All inference runs locally in the browser:

- `models/silero_vad.onnx` detects speech chunks.
- `models/smart-turn-v3.2-cpu.onnx` predicts whether the turn is complete.
- `vendor/ort.min.js` and the WASM files provide the local ONNX Runtime Web WASM fallback.
- `app.js` includes a built-in Whisper-compatible feature extractor and prefers ONNX Runtime Web's WebGPU build when available, then falls back to WASM if WebGPU is unavailable or cannot run the models.

Implementation notes for the microphone pipeline, VAD I/O, smart-turn I/O, and audio buffer handling live in [audio-pipeline.md](./audio-pipeline.md).

Use an HTTPS origin for microphone access. GitHub Pages satisfies this requirement.

## Cache updates on GitHub Pages

GitHub Pages can keep previously loaded HTML and assets briefly cached. `index.html` loads CSS and JavaScript with a `?v=` value generated from the current Unix time, so those assets are revalidated on each page load. When the HTML structure or model files change, update the `app-version` meta value in `index.html`; the page checks the latest `index.html` with `cache: "no-store"` and reloads with a versioned URL when it detects a newer version.
