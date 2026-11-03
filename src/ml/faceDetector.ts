/**
 * STEP 1: face detection + crop (the biggest accuracy win).
 *
 * A full photo is mostly background, clothes and lighting; MobileNet features
 * of the whole frame describe those as much as the person. Cropping to the
 * face first makes every sample "the same kind of picture", so the head learns
 * identity rather than scenery.
 *
 * Detector: MediaPipe BlazeFace (short range) via @tensorflow-models/face-detection,
 * tfjs runtime, model served from public/models/blazeface.
 */
import * as faceDetection from '@tensorflow-models/face-detection';
import { IMAGE_SIZE, initBackend } from './mobilenet';
import type { FaceBox } from '../types';

export const BLAZEFACE_URL = `${import.meta.env.BASE_URL}models/blazeface/model.json`;

/** Crop = face box grown by 20% of its size on every side, then squared. */
export const FACE_MARGIN = 0.2;

/**
 * Background faces much smaller than the main one (a poster, someone far
 * behind) don't count towards "multiple faces".
 */
export const SECONDARY_FACE_MIN_RATIO = 0.45;

/** Out-of-frame parts of a crop are filled with mid-gray (also used by augmentation). */
export const FILL_GRAY = 128;

export type Detectable = HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageBitmap;

let detectorPromise: Promise<faceDetection.FaceDetector> | null = null;

export function loadFaceDetector(): Promise<faceDetection.FaceDetector> {
  detectorPromise ??= (async () => {
    await initBackend();
    let detector: faceDetection.FaceDetector;
    try {
      detector = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
        runtime: 'tfjs',
        modelType: 'short',
        maxFaces: 5, // need more than 1 to notice multi-face photos
        detectorModelUrl: BLAZEFACE_URL,
      });
    } catch (err) {
      throw new Error(
        `Could not load the face detector from ${BLAZEFACE_URL}. Run "npm run fetch-models". (${err instanceof Error ? err.message : err})`,
      );
    }
    // Warm-up compiles shaders so the first real photo isn't slow.
    const blank = document.createElement('canvas');
    blank.width = blank.height = 128;
    await detector.estimateFaces(blank);
    return detector;
  })().catch((err) => {
    detectorPromise = null;
    throw err;
  });
  return detectorPromise;
}

function sourceSize(source: Detectable) {
  if (source instanceof HTMLVideoElement) return { width: source.videoWidth, height: source.videoHeight };
  if (source instanceof HTMLImageElement) return { width: source.naturalWidth, height: source.naturalHeight };
  return { width: source.width, height: source.height };
}

/** All faces, largest first. */
export async function detectFaces(source: Detectable): Promise<FaceBox[]> {
  const detector = await loadFaceDetector();
  const faces = await detector.estimateFaces(source, { flipHorizontal: false });
  return faces
    .map(({ box }) => ({ xMin: box.xMin, yMin: box.yMin, width: box.width, height: box.height }))
    .filter((b) => b.width > 0 && b.height > 0)
    .sort((a, b) => b.width * b.height - a.width * a.height);
}

/** Faces that matter: the largest, plus any at least SECONDARY_FACE_MIN_RATIO of its width. */
export function significantFaces(faces: FaceBox[]): FaceBox[] {
  if (faces.length === 0) return faces;
  const main = faces[0].width;
  return faces.filter((f) => f.width >= main * SECONDARY_FACE_MIN_RATIO);
}

/** The square crop region (in source pixels) for a face box: 20% margin per side. */
export function cropRegion(box: FaceBox) {
  const side = Math.max(box.width, box.height) * (1 + 2 * FACE_MARGIN);
  const cx = box.xMin + box.width / 2;
  const cy = box.yMin + box.height / 2;
  return { sx: cx - side / 2, sy: cy - side / 2, side };
}

/** Draws the face crop into a 224x224 canvas (gray where the margin leaves the frame). */
export function cropFace(source: Detectable, box: FaceBox, canvas = document.createElement('canvas')): HTMLCanvasElement {
  const { sx, sy, side } = cropRegion(box);
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = `rgb(${FILL_GRAY},${FILL_GRAY},${FILL_GRAY})`;
  ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  // The source rect may extend past the image; drawImage clips it and keeps the scale.
  ctx.drawImage(source, sx, sy, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return canvas;
}

export type FaceCropResult =
  | { status: 'ok'; crop: HTMLCanvasElement; box: FaceBox; faceCount: number }
  | { status: 'none' }
  | { status: 'multiple'; count: number; boxes: FaceBox[] };

/**
 * Detects faces and crops one.
 *  - mode 'strict' (training data): exactly one significant face, otherwise rejected.
 *  - mode 'largest' (inference): classify the largest face, report how many there were.
 */
export async function detectAndCropFace(
  source: Detectable,
  mode: 'strict' | 'largest' = 'strict',
  canvas?: HTMLCanvasElement,
): Promise<FaceCropResult> {
  const { width, height } = sourceSize(source);
  if (!width || !height) return { status: 'none' };
  const faces = significantFaces(await detectFaces(source));
  if (faces.length === 0) return { status: 'none' };
  if (faces.length > 1 && mode === 'strict') return { status: 'multiple', count: faces.length, boxes: faces };
  return { status: 'ok', crop: cropFace(source, faces[0], canvas), box: faces[0], faceCount: faces.length };
}
