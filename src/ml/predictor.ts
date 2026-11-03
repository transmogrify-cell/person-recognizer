/**
 * STEP 6: inference with confidence gating.
 *
 *   image -> BlazeFace -> largest face crop -> MobileNet features -> head -> softmax
 *         -> top class; if it is the "Unknown" class, or its confidence is below
 *            the threshold, the answer is "Unknown / Not sure".
 *
 * Softmax always sums to 1 across the trained classes, so a stranger still gets
 * *some* top match. The gate and the Unknown class are what let the model
 * decline instead of force-matching.
 */
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { NOT_SURE_LABEL } from '../config';
import { detectAndCropFace, type Detectable, type FaceCropResult } from './faceDetector';
import { extractFeatures, loadMobilenet, type FeatureExtractor } from './mobilenet';
import { argmax } from './evaluator';
import { closeImage, decodeFile } from '../utils/image';
import type { ClassKind, FaceBox, ModelMetadata, Prediction } from '../types';

/** Class probabilities for a 224x224 face crop. All intermediates are freed by tf.tidy. */
export async function classifyCrop(extractor: FeatureExtractor, head: tf.LayersModel, crop: HTMLCanvasElement): Promise<Float32Array> {
  const probs = tf.tidy(() => {
    const features = extractFeatures(extractor, tf.browser.fromPixels(crop, 3));
    return head.predict(features) as tf.Tensor2D;
  });
  try {
    return (await probs.data()) as Float32Array;
  } finally {
    probs.dispose();
  }
}

/** Applies the Unknown class and the confidence threshold to raw probabilities. */
export function predictWithConfidenceThreshold(
  probs: ArrayLike<number>,
  labels: string[],
  kinds: ClassKind[],
  threshold: number,
): Prediction {
  const topIndex = argmax(probs);
  const confidence = probs[topIndex];
  const allProbs: Record<string, number> = {};
  labels.forEach((l, i) => (allProbs[l] = probs[i]));

  if (kinds[topIndex] === 'unknown') {
    return { person: labels[topIndex], confidence, allProbs, topIndex, isUnknown: true, reason: 'unknown-class' };
  }
  if (confidence < threshold) {
    return { person: NOT_SURE_LABEL, confidence, allProbs, topIndex, isUnknown: true, reason: 'below-threshold' };
  }
  return { person: labels[topIndex], confidence, allProbs, topIndex, isUnknown: false, reason: 'match' };
}

export interface ImagePrediction {
  face: FaceCropResult['status'];
  box?: FaceBox;
  faceCount: number;
  prediction?: Prediction;
}

/** Full pipeline for one image: detect -> crop the largest face -> classify -> gate. */
export async function predictImage(
  source: Detectable,
  opts: { head: tf.LayersModel; labels: string[]; kinds: ClassKind[]; threshold: number },
): Promise<ImagePrediction> {
  const face = await detectAndCropFace(source, 'largest');
  if (face.status !== 'ok') return { face: face.status, faceCount: face.status === 'multiple' ? face.count : 0 };
  const extractor = await loadMobilenet();
  const probs = await classifyCrop(extractor, opts.head, face.crop);
  return {
    face: 'ok',
    box: face.box,
    faceCount: face.faceCount,
    prediction: predictWithConfidenceThreshold(probs, opts.labels, opts.kinds, opts.threshold),
  };
}

// ───────────────────────────── loadModel() ─────────────────────────────

export interface LoadedPrediction {
  /** null when no face was found. */
  person: string | null;
  confidence: number;
  allProbs: Record<string, number>;
  faceCount: number;
  box?: FaceBox;
}

export interface LoadedModel {
  labels: string[];
  kinds: ClassKind[];
  metadata: ModelMetadata;
  head: tf.LayersModel;
  /** Accepts a File/Blob or an image/video/canvas. Uses metadata.confidenceThreshold unless overridden. */
  predict(image: Blob | Detectable, threshold?: number): Promise<LoadedPrediction>;
  dispose(): void;
}

/** A hosted folder, the exported .zip, or the individual files (model.json, weights.bin, metadata.json). */
export type ModelSource = { modelUrl: string; metadataUrl?: string } | File | File[] | FileList;

/**
 * Reloads an exported model: loads BlazeFace + MobileNet, loads the head and
 * metadata, checks they fit together, and returns predict(imageFile).
 */
export async function loadModel(source: ModelSource): Promise<LoadedModel> {
  const extractor = await loadMobilenet();
  const { head, metadata } = await loadHead(source);

  const inputDim = head.inputs[0].shape[1];
  const outputDim = head.outputs[0].shape[1];
  if (inputDim !== extractor.featureDim || outputDim !== metadata.labels.length) {
    head.dispose();
    throw new Error(
      `Model doesn't match: head ${inputDim}->${outputDim}, MobileNet gives ${extractor.featureDim}-d features, metadata has ${metadata.labels.length} labels.`,
    );
  }
  const kinds: ClassKind[] = metadata.labels.map((l) => (l === metadata.unknownLabel ? 'unknown' : 'person'));

  return {
    labels: metadata.labels,
    kinds,
    metadata,
    head,
    async predict(image, threshold = metadata.confidenceThreshold) {
      const decoded = image instanceof Blob ? await decodeFile(image) : image;
      try {
        const result = await predictImage(decoded, { head, labels: metadata.labels, kinds, threshold });
        if (!result.prediction) return { person: null, confidence: 0, allProbs: {}, faceCount: result.faceCount };
        const { person, confidence, allProbs } = result.prediction;
        return { person, confidence, allProbs, faceCount: result.faceCount, box: result.box };
      } finally {
        if (image instanceof Blob) closeImage(decoded as ImageBitmap | HTMLImageElement);
      }
    },
    dispose: () => head.dispose(),
  };
}

async function loadHead(source: ModelSource): Promise<{ head: tf.LayersModel; metadata: ModelMetadata }> {
  if (!(source instanceof File) && !Array.isArray(source) && !(source instanceof FileList)) {
    const metadataUrl = source.metadataUrl ?? new URL('metadata.json', new URL(source.modelUrl, location.href)).href;
    const [head, metadata] = await Promise.all([
      tf.loadLayersModel(source.modelUrl),
      fetch(metadataUrl).then((r) => {
        if (!r.ok) throw new Error(`metadata.json: HTTP ${r.status}`);
        return r.json() as Promise<ModelMetadata>;
      }),
    ]);
    return { head, metadata };
  }
  if (source instanceof File && /\.zip$/i.test(source.name)) {
    const zip = await JSZip.loadAsync(source);
    const files = await Promise.all(
      Object.values(zip.files)
        .filter((f) => !f.dir)
        .map(async (f) => new File([await f.async('arraybuffer')], f.name.split('/').pop()!)),
    );
    return loadHeadFromFiles(files);
  }
  return loadHeadFromFiles(source instanceof File ? [source] : Array.from(source));
}

async function loadHeadFromFiles(files: File[]) {
  const byName = new Map(files.map((f) => [f.name, f]));
  const modelFile = byName.get('model.json');
  const metadataFile = byName.get('metadata.json');
  if (!modelFile) throw new Error('model.json is missing.');
  if (!metadataFile) throw new Error('metadata.json is missing.');
  const modelJson = JSON.parse(await modelFile.text()) as tf.io.ModelJSON;
  const metadata = JSON.parse(await metadataFile.text()) as ModelMetadata;

  const buffers: ArrayBuffer[] = [];
  for (const group of modelJson.weightsManifest) {
    for (const path of group.paths) {
      const shard = byName.get(path.split('/').pop()!);
      if (!shard) throw new Error(`Weights file "${path}" is missing.`);
      buffers.push(await shard.arrayBuffer());
    }
  }
  const head = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs: modelJson.weightsManifest.flatMap((g) => g.weights),
      weightData: tf.io.concatenateArrayBuffers(buffers),
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
    }),
  );
  return { head, metadata };
}
