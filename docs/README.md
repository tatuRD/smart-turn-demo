# Smart Turn Browser Demo

This directory is ready to publish with GitHub Pages.

All inference runs locally in the browser:

- `models/silero_vad.onnx` detects speech chunks.
- `models/smart-turn-v3.2-cpu.onnx` predicts whether the turn is complete.
- `vendor/ort.min.js` and the WASM files provide the local ONNX Runtime Web WASM fallback.
- When available, the demo dynamically loads Transformers.js for Whisper feature extraction and ONNX Runtime Web's WebGPU build from jsDelivr, then falls back to WASM if WebGPU is unavailable or cannot run the models.

Use an HTTPS origin for microphone access. GitHub Pages satisfies this requirement.
