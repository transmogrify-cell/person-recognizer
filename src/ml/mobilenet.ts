/**
 * STEP 2: frozen MobileNet V2 feature extractor.
 *
 * MobileNet V2 (width 1.0, 224x224) was trained on ImageNet. We drop its
 * 1000-way classifier and read the layer just before it: the 1280-d global
 * average pool ("penultimate layer"). Those 1280 numbers describe a face crop
 * well enough for a tiny head to tell your people apart.
 *
 * Frozen by construction: it is a tfjs GraphModel (weights are graph
 * constants), and we only ever call inference on it.
 *
 * Preprocessing: this TF-Hub graph takes pixels in [0, 1] and rescales them to
 * MobileNet's native [-1, 1] range internally. The package only knows that when
 * it downloads the model itself, so we pass inputRange: [0, 1] explicitly.
 */
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

export const IMAGE_SIZE = 224;
export const MOBILENET_URL = `${import.meta.env.BASE_URL}models/mobilenet/model.json`;
export const MOBILENET_INPUT_RANGE: [number, number] = [0, 1];
export const EMBEDDING_LAYER = 'module_apply_default/MobilenetV2/Logits/AvgPool';
export const MOBILENET_SOURCE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2';

export interface FeatureExtractor {
  model: mobilenet.MobileNet;
  featureDim: number;
}

let backendPromise: Promise<string> | null = null;

/** Prefer WebGL (GPU); fall back to the CPU backend if it fails. Shared by both models. */
export function initBackend(): Promise<string> {
  backendPromise ??= (async () => {
    try {
      if (await tf.setBackend('webgl')) {
        await tf.ready();
        tf.tidy(() => tf.add(tf.scalar(1), tf.scalar(1)).dataSync());
        return 'webgl';
      }
    } catch (err) {
      console.warn('[tfjs] WebGL backend failed, falling back to CPU', err);
    }
    await tf.setBackend('cpu');
    await tf.ready();
    return 'cpu';
  })();
  return backendPromise;
}

let extractorPromise: Promise<FeatureExtractor> | null = null;

/** Loads MobileNet once (from public/models/mobilenet) and returns the feature extractor. */
export function loadMobilenet(): Promise<FeatureExtractor> {
  extractorPromise ??= (async () => {
    await initBackend();
    let model: mobilenet.MobileNet;
    try {
      model = await mobilenet.load({ version: 2, alpha: 1.0, modelUrl: MOBILENET_URL, inputRange: MOBILENET_INPUT_RANGE });
    } catch (err) {
      throw new Error(
        `Could not load MobileNet from ${MOBILENET_URL}. Run "npm run fetch-models". (${err instanceof Error ? err.message : err})`,
      );
    }
    // Probe the embedding size (and compile the shaders for that path).
    const probe = model.infer(tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3]), true);
    const featureDim = probe.shape[probe.shape.length - 1] as number;
    probe.dispose();
    return { model, featureDim };
  })().catch((err) => {
    extractorPromise = null;
    throw err;
  });
  return extractorPromise;
}

/**
 * Pixels -> features. `pixels` is [n, 224, 224, 3] (or a single [224, 224, 3])
 * with raw values in [0, 255]; the package scales them to the graph's input
 * range. Returns [n, 1280]. The caller owns both tensors.
 */
export function extractFeatures(extractor: FeatureExtractor, pixels: tf.Tensor3D | tf.Tensor4D): tf.Tensor2D {
  return extractor.model.infer(pixels, true) as tf.Tensor2D;
}
