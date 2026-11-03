/**
 * STEP 3: data augmentation, done with tf ops on the GPU before feature extraction.
 *
 * Each face crop becomes 6 variants:
 *   0 original · 1 horizontal flip · 2 brightness ±15% · 3 rotation ±10° · 4 zoom 0.9x · 5 zoom 1.1x
 *
 * Brightness and rotation alternate direction per photo (+ for one, − for the
 * next, decided by a hash of the sample id), so a class gets both directions
 * while each photo still yields exactly 6 variants. Being deterministic, the
 * variants (and their cached features) are the same on every retrain.
 */
import * as tf from '@tensorflow/tfjs';
import { FILL_GRAY } from './faceDetector';

export const VARIANT_NAMES = ['original', 'flip', 'brightness', 'rotate', 'zoom 0.9x', 'zoom 1.1x'] as const;
export const VARIANTS_PER_IMAGE = VARIANT_NAMES.length;

const BRIGHTNESS = 0.15;
const ROTATION_RAD = (10 * Math.PI) / 180;
const ZOOM_OUT = 0.9;
const ZOOM_IN = 1.1;

/** Stable small hash so a sample always gets the same augmentation directions. */
export function seedFromId(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return h >>> 0;
}

/**
 * pixels: [224, 224, 3] float in [0, 255]  ->  [6, 224, 224, 3] float in [0, 255].
 * Caller owns the result.
 */
export function generateAugmentedVariants(pixels: tf.Tensor3D, seed: number): tf.Tensor4D {
  return tf.tidy(() => {
    const [h, w] = pixels.shape;
    const x = tf.expandDims(tf.cast(pixels, 'float32'), 0) as tf.Tensor4D;
    const sign = seed % 2 === 0 ? 1 : -1;
    const sign2 = (seed >>> 1) % 2 === 0 ? 1 : -1;

    const flip = tf.image.flipLeftRight(x);
    const bright = tf.clipByValue(tf.mul(x, 1 + sign * BRIGHTNESS), 0, 255) as tf.Tensor4D;
    const rotated = tf.image.rotateWithOffset(x, sign2 * ROTATION_RAD, FILL_GRAY);

    // Zoom = crop a centered box and resize back. Boxes are [y1, x1, y2, x2] in
    // normalized coordinates; a box larger than the image (zoom out) pads with gray.
    const zoomBox = (factor: number) => {
      const half = 0.5 / factor;
      return [0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half];
    };
    const zoomed = tf.image.cropAndResize(
      tf.concat([x, x], 0) as tf.Tensor4D,
      [zoomBox(ZOOM_OUT), zoomBox(ZOOM_IN)],
      [0, 1],
      [h, w],
      'bilinear',
      FILL_GRAY,
    );

    return tf.concat([x, flip, bright, rotated, zoomed], 0) as tf.Tensor4D;
  });
}
