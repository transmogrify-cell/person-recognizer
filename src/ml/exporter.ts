/**
 * Export. Only the trained head is saved; BlazeFace and MobileNet are the same
 * for every project and are reloaded at load time (see loadModel()).
 *
 *  - TensorFlow.js zip: model.json + weights.bin (head) + metadata.json, plus
 *    convert_to_keras.py so the same zip also yields a Keras .h5.
 *  - Keras: the browser can't write HDF5, so a Python snippet reads the TF.js
 *    weights.bin directly and rebuilds the head in tf.keras.
 */
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { EMBEDDING_LAYER, IMAGE_SIZE, MOBILENET_INPUT_RANGE, MOBILENET_SOURCE } from './mobilenet';
import { FACE_MARGIN } from './faceDetector';
import { slugify, triggerDownload } from '../utils/download';
import type { ModelMetadata, TrainedModel } from '../types';

export function buildMetadata(trained: TrainedModel, labels: string[], threshold: number): ModelMetadata {
  const unknownIndex = trained.kinds.indexOf('unknown');
  return {
    labels,
    imageSize: IMAGE_SIZE,
    normalization: '[-1,1]',
    baseModel: 'mobilenet_v2',
    faceCrop: true,
    confidenceThreshold: threshold,
    unknownLabel: unknownIndex >= 0 ? labels[unknownIndex] : null,
    faceDetector: 'mediapipe_blazeface_short',
    faceMargin: FACE_MARGIN,
    featureDim: trained.featureDim,
    mobilenet: {
      version: 2,
      alpha: 1,
      inputRange: MOBILENET_INPUT_RANGE,
      embeddingLayer: EMBEDDING_LAYER,
      source: MOBILENET_SOURCE,
    },
    createdAt: new Date(trained.trainedAt).toISOString(),
    generator: 'person-recognizer',
    version: 1,
  };
}

async function headArtifacts(head: tf.LayersModel): Promise<tf.io.ModelArtifacts> {
  let captured: tf.io.ModelArtifacts | null = null;
  await head.save(
    tf.io.withSaveHandler(async (artifacts) => {
      captured = artifacts;
      return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
    }),
  );
  if (!captured) throw new Error('Saving the model produced no data.');
  return captured;
}

export async function buildTfjsZip(trained: TrainedModel, labels: string[], threshold: number): Promise<Blob> {
  const artifacts = await headArtifacts(trained.head);
  const weightData = artifacts.weightData;
  const modelJson: tf.io.ModelJSON = {
    modelTopology: artifacts.modelTopology!,
    format: artifacts.format ?? 'layers-model',
    generatedBy: artifacts.generatedBy,
    convertedBy: null,
    weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs ?? [] }],
  };
  const zip = new JSZip();
  zip.file('model.json', JSON.stringify(modelJson, null, 2));
  zip.file('weights.bin', Array.isArray(weightData) ? tf.io.concatenateArrayBuffers(weightData) : weightData ?? new ArrayBuffer(0));
  zip.file('metadata.json', JSON.stringify(buildMetadata(trained, labels, threshold), null, 2));
  zip.file('convert_to_keras.py', kerasPythonSnippet());
  return zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
}

export async function exportTfjs(trained: TrainedModel, labels: string[], threshold: number, name = 'person-model'): Promise<void> {
  triggerDownload(await buildTfjsZip(trained, labels, threshold), `${slugify(name)}-tfjs.zip`);
}

/**
 * Python that rebuilds the head in tf.keras from the TF.js export and saves .h5.
 * It parses model.json's weight manifest and slices weights.bin with numpy, so
 * it needs nothing but TensorFlow.
 */
export function kerasPythonSnippet(): string {
  return `#!/usr/bin/env python3
"""
Rebuild the person-recognition HEAD exported from the browser as a Keras .h5.

Put this file next to model.json, weights.bin and metadata.json (the unzipped
TF.js export) and run:

    pip install "tensorflow>=2.12"
    python convert_to_keras.py            # -> person_head.h5 + labels.txt

IMPORTANT: this is only the classification head. Its input is the 1280-d
MobileNet V2 feature vector of a FACE CROP, not an image. To classify a photo
you must reproduce the browser pipeline:
  1. detect the face (BlazeFace / MediaPipe face detection),
  2. crop a square around it with a 20% margin per side, resize to 224x224,
  3. get the 1280-d features from MobileNet V2 1.0 224 (TF-Hub
     "google/imagenet/mobilenet_v2_100_224", pixels scaled to [0, 1]),
  4. head.predict(features) -> probabilities in labels.txt order.
"""
import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
L2 = tf.keras.regularizers.l2(0.001)


def load_tfjs_weights(folder):
    """name -> numpy array, read from the TF.js weights manifest."""
    with open(os.path.join(folder, "model.json"), encoding="utf-8") as f:
        manifest = json.load(f)["weightsManifest"]
    weights = {}
    for group in manifest:
        data = b"".join(open(os.path.join(folder, p), "rb").read() for p in group["paths"])
        offset = 0
        for spec in group["weights"]:
            count = int(np.prod(spec["shape"])) if spec["shape"] else 1
            weights[spec["name"]] = np.frombuffer(data, np.float32, count, offset).reshape(spec["shape"])
            offset += count * 4
    return weights


def build_head(feature_dim, num_classes):
    """Same architecture as the browser head, with the same layer names."""
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(feature_dim,), name="mobilenet_features"),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=L2, name="head_dense_1"),
            tf.keras.layers.BatchNormalization(name="head_bn"),
            tf.keras.layers.Dropout(0.4, name="head_dropout_1"),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=L2, name="head_dense_2"),
            tf.keras.layers.Dropout(0.3, name="head_dropout_2"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="head_output"),
        ],
        name="person_head",
    )


def main():
    with open(os.path.join(HERE, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    labels = meta["labels"]
    w = load_tfjs_weights(HERE)
    head = build_head(meta["featureDim"], len(labels))

    head.get_layer("head_dense_1").set_weights([w["head_dense_1/kernel"], w["head_dense_1/bias"]])
    head.get_layer("head_bn").set_weights(
        [w["head_bn/gamma"], w["head_bn/beta"], w["head_bn/moving_mean"], w["head_bn/moving_variance"]]
    )
    head.get_layer("head_dense_2").set_weights([w["head_dense_2/kernel"], w["head_dense_2/bias"]])
    head.get_layer("head_output").set_weights([w["head_output/kernel"], w["head_output/bias"]])

    head.save(os.path.join(HERE, "person_head.h5"))
    with open(os.path.join(HERE, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(labels) + "\\n")
    print(f"Saved person_head.h5 ({len(labels)} classes: {', '.join(labels)})")
    print(f"Unknown class: {meta.get('unknownLabel')}; suggested confidence threshold: {meta['confidenceThreshold']}")


if __name__ == "__main__":
    main()
`;
}
