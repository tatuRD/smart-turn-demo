# Smart Turn Browser Demo

This directory is ready to publish with GitHub Pages.

All inference runs locally in the browser:

- `models/silero_vad.onnx` detects speech chunks.
- `models/smart-turn-v3.2-cpu.onnx` predicts whether the turn is complete.
- `vendor/ort.min.js` and the WASM files provide ONNX Runtime Web 1.18.0 CPU execution.

Use an HTTPS origin for microphone access. GitHub Pages satisfies this requirement.
