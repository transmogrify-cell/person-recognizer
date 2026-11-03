// Product limits and defaults. Model constants live in src/ml/*.

/** Up to 8 people plus the optional "Unknown" class (one categorical color each). */
export const MAX_PERSONS = 8;

/** Minimum samples before Train unlocks, and what counts as "Good". */
export const MIN_SAMPLES_PERSON = 20;
export const GOOD_SAMPLES_PERSON = 40;
export const MIN_SAMPLES_UNKNOWN = 10;
export const GOOD_SAMPLES_UNKNOWN = 20;

/** Hard cap per class so a long webcam hold can't exhaust memory. */
export const MAX_SAMPLES_PER_CLASS = 500;

/** Webcam "Hold to capture" rate (each frame also runs the face detector). */
export const CAPTURE_FPS = 10;

/** Name of the special rejection class. */
export const UNKNOWN_LABEL = 'Unknown';

/** Shown when the top confidence is below the threshold. */
export const NOT_SURE_LABEL = 'Unknown / Not sure';

export const DEFAULT_THRESHOLD = 0.6;
export const THRESHOLD_MIN = 0.3;
export const THRESHOLD_MAX = 0.9;

/** Stored face crops: 224x224 JPEG. */
export const SAMPLE_MIME = 'image/jpeg';
export const SAMPLE_QUALITY = 0.92;

/**
 * Photos pushed through MobileNet per step. Each photo becomes 6 augmented
 * variants, so one step is a [24, 224, 224, 3] batch.
 */
export const PHOTOS_PER_BATCH = 4;

/** A per-class precision/recall below this gets a "needs more data" warning. */
export const WEAK_CLASS_THRESHOLD = 0.7;

export const DEFAULT_TRAINING = {
  epochs: 40,
  batchSize: 16,
  learningRate: 0.0005,
  validationSplit: 0.2,
  patience: 5,
};

/**
 * Categorical identity colors (validated palette: CVD-safe in this order).
 * Slot i goes to the i-th person; "Unknown" is always neutral gray.
 */
export const PERSON_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
export const UNKNOWN_COLOR = '#8a8984';
