// @tensorflow-models/face-detection statically imports @mediapipe/face_detection
// for its optional "mediapipe" runtime (a WASM build fetched from a CDN). This
// app only uses the "tfjs" runtime, so vite.config.ts aliases the import here:
// no CDN code is bundled, and choosing the mediapipe runtime fails loudly.
export class FaceDetection {
  constructor() {
    throw new Error('The MediaPipe runtime is not bundled; use runtime: "tfjs".');
  }
}
