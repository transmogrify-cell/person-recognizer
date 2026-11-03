import type * as tf from '@tensorflow/tfjs';

/** A real person, or the special class that teaches the model to say "I don't know". */
export type ClassKind = 'person' | 'unknown';

/** One training photo: a 224x224 face crop (20% margin), stored as a JPEG data URL. */
export interface Sample {
  id: string;
  dataUrl: string;
  source: 'webcam' | 'upload';
}

export interface Person {
  id: string;
  name: string;
  kind: ClassKind;
  samples: Sample[];
}

export interface TrainingParams {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  /** Early stopping: epochs without a val_loss improvement before stopping. */
  patience: number;
}

export interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  valLoss?: number;
  valAcc?: number;
}

export type TrainingPhase = 'idle' | 'extracting' | 'training' | 'evaluating' | 'done' | 'error';

export interface TrainingProgress {
  phase: TrainingPhase;
  /** Feature extraction counts augmented images (photos x 6). */
  extracted: number;
  extractTotal: number;
  photos: number;
  epoch: number;
  epochFraction: number;
  totalEpochs: number;
  history: EpochLog[];
  /** Epoch (1-based) whose weights were kept (lowest val_loss). */
  bestEpoch?: number;
  stoppedEarly?: boolean;
  error?: string;
}

export interface ClassMetrics {
  label: string;
  kind: ClassKind;
  /** Of the validation photos predicted as this class, the fraction that really were. NaN if none predicted. */
  precision: number;
  /** Of this class's validation photos, the fraction recognised correctly. NaN if none held out. */
  recall: number;
  f1: number;
  /** Validation photos of this class. */
  support: number;
  /** All photos of this class (train + validation). */
  samples: number;
}

/** Post-training evaluation on the held-out validation photos (original, un-augmented). */
export interface Evaluation {
  labels: string[];
  overallAccuracy: number;
  perClass: ClassMetrics[];
  /** confusion[actual][predicted] = count. */
  confusion: number[][];
  /** Softmax output per validation photo, for threshold analysis. */
  valProbs: number[][];
  valTrue: number[];
}

/** The trained head plus everything needed to interpret its outputs. MobileNet is not part of it. */
export interface TrainedModel {
  head: tf.LayersModel;
  /** Class ids in output order (so renames after training still map). */
  classIds: string[];
  labels: string[];
  kinds: ClassKind[];
  featureDim: number;
  trainedAt: number;
  sampleCount: number;
  evaluation: Evaluation | null;
  bestEpoch?: number;
  epochsRun?: number;
}

/** metadata.json in the TF.js export. */
export interface ModelMetadata {
  labels: string[];
  imageSize: number;
  normalization: '[-1,1]';
  baseModel: 'mobilenet_v2';
  faceCrop: boolean;
  confidenceThreshold: number;
  /** Label of the rejection class, or null if the model has none. */
  unknownLabel: string | null;
  faceDetector: 'mediapipe_blazeface_short';
  /** Crop = face box enlarged by this fraction on every side, squared, resized to imageSize. */
  faceMargin: number;
  featureDim: number;
  mobilenet: {
    version: 2;
    alpha: 1;
    /** Pixel range fed to the TF-Hub graph; it rescales to [-1, 1] internally. */
    inputRange: [number, number];
    embeddingLayer: string;
    source: string;
  };
  createdAt: string;
  generator: string;
  version: number;
}

export interface Prediction {
  /** Display label: a person, "Unknown", or "Unknown / Not sure". */
  person: string;
  confidence: number;
  /** label -> probability, in output order. */
  allProbs: Record<string, number>;
  /** Top class index before gating. */
  topIndex: number;
  isUnknown: boolean;
  reason: 'match' | 'unknown-class' | 'below-threshold';
}

export interface FaceBox {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
}
