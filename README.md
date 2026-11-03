# Person Recognizer: a browser-only person recognition trainer

A Teachable-Machine-style web app for recognising **the specific people you choose**. You add a class per person ("Alice", "Bob", "Me") and an optional **Unknown** class. Then you upload photos or capture from the webcam, train in the browser, and get an **Accuracy Report**: overall accuracy, per-person precision and recall, and a confusion matrix. A live preview tells you who is in front of the camera, and answers "Unknown / Not sure" when it isn't confident.

Everything runs locally in the browser tab: face detection, feature extraction, training, evaluation and prediction. There is no server, no cloud call and no telemetry. Both pre-trained models are stored in this repo (`public/models/`), so the app works fully offline after `npm install`.

> **This is a transfer-learning classifier, not face recognition.** It is a frozen-feature + softmax model, not FaceNet/ArcFace embeddings. It recognises the people it was trained on and nobody else, and it is a learning/demo tool. **Do not use it for security or authentication.**

---

## Contents

1. [Quick start](#quick-start)
2. [Using the app](#using-the-app)
3. [How it works](#how-it-works)
   - [The pipeline at a glance](#the-pipeline-at-a-glance)
   - [Step 1: Face detection and crop](#step-1-face-detection-and-crop)
   - [Step 2: Frozen MobileNet V2 feature extractor](#step-2-frozen-mobilenet-v2-feature-extractor)
   - [Step 3: Data augmentation (6× per photo)](#step-3-data-augmentation-6-per-photo)
   - [Step 4: The classification head and how it's trained](#step-4-the-classification-head-and-how-its-trained)
   - [Step 5: The Accuracy Report](#step-5-the-accuracy-report)
   - [Step 6: Inference with confidence gating](#step-6-inference-with-confidence-gating)
   - [Step 7: The "Unknown" class](#step-7-the-unknown-class)
   - [Why transfer learning works here (and where it doesn't)](#why-transfer-learning-works-here-and-where-it-doesnt)
   - [Performance and memory](#performance-and-memory)
4. [Data requirements](#data-requirements)
5. [Export formats](#export-formats)
6. [Project structure](#project-structure)
7. [Demo test results](#demo-test-results)
8. [Error handling](#error-handling)
9. [Limitations](#limitations)
10. [Design decisions and deviations from the spec](#design-decisions-and-deviations-from-the-spec)

---

## Quick start

```bash
npm install
npm run dev          # http://localhost:5174
```

Requirements: Node 20.19+ and a recent Chrome, Edge, Firefox or Safari. A GPU with WebGL is strongly recommended. The webcam needs a secure origin, and `localhost` counts as one.

```bash
npm run build        # type-check + static production build in dist/
npm run preview      # serve dist/
npm run fetch-models -- --force   # re-download the two models (already committed)
```

## Using the app

| Screen | What you do |
| --- | --- |
| **1. Start** | Click **Person Recognition Trainer**, or **Open an exported model** to jump straight to the preview. |
| **2. Collect** | One card per person. Rename it, then **Hold to capture** from the webcam (about 10 photos per second while held) or **Upload photos** (multi-select or drag and drop). Every photo is face-detected and cropped; the gallery shows the face crops. Each card has a sample count and a quality badge (Good / Needs more / Too few). Add the suggested **Unknown** card and fill it with other people. |
| **3. Train** | **Train Model** unlocks when every person has at least 20 photos and Unknown (if used) has at least 10. You'll see "Extracting features: 45/240 images", then "Training: epoch 7/40" with live loss and accuracy charts. Training stops early when validation loss stops improving. |
| **4. Report** | Overall accuracy, a per-person table (precision, recall, F1, test count, sample count), a confusion matrix heatmap, warnings for weak classes, and what the confidence threshold would do to the test photos. |
| **5. Preview** | Live webcam with a face box and name overlay, or **Snapshot test** with an uploaded photo. Shows the predicted person in large text, a confidence bar per class, and a **threshold slider** (0.3–0.9) that turns low-confidence answers into "Unknown / Not sure". **Export Model** is here. |

**Advanced** (bottom bar on the Collect screen) sets max epochs (40), early-stopping patience (5), batch size (16) and learning rate (0.0005).

---

## How it works

### The pipeline at a glance

```
           ┌─────────────── frozen, pre-trained ────────────────┐   ┌──────────── trained in your browser ────────────┐
 photo ──► BlazeFace ──► 1 face? ──► square crop +20% ──► 224×224 ──► MobileNet V2 ──► 1280 numbers ──► small head ──► softmax
           (detector)    else skip    margin per side      face        (features)      per face          (Dense…)      per class
                                                            │
                               training only: 6 augmented variants (flip, brightness, rotate, zoom ×2)
```

Two pre-trained networks do the heavy lifting, and neither is ever modified:

- **BlazeFace** finds the face.
- **MobileNet V2** turns the face crop into a 1280-number description.

The only part that learns is a **small head** of about 360k parameters, which maps those 1280 numbers to your classes. Because MobileNet runs only once per image and the head is tiny, training takes seconds.

Source files: `src/ml/faceDetector.ts` → `mobilenet.ts` → `augmentation.ts` → `trainer.ts` → `evaluator.ts` → `predictor.ts` → `exporter.ts`.

---

### Step 1: Face detection and crop

**Why this matters.** A full photo is mostly background, clothes and lighting. MobileNet's features of the whole frame describe those things as much as the person, so two photos of Alice in different rooms can look less alike than Alice and Bob in the same room. Cropping to the face first means every sample is "the same kind of picture", and the head learns identity instead of scenery. This is the single biggest accuracy gain in the pipeline.

**Detector.** MediaPipe **BlazeFace (short range)** through `@tensorflow-models/face-detection`, using the `tfjs` runtime:

- The model is served from `public/models/blazeface/`.
- It looks at a 128×128 version of the image and predicts boxes from 896 anchors.
- Detections need a score of at least 0.5.
- Overlapping boxes are merged with non-max suppression (IoU 0.3).
- It is set to return up to 5 faces, so the app can tell when a photo contains more than one.
- The "short range" model is tuned for faces within about 2 m of the camera, which suits selfies, webcams and portraits.

(The package's alternative `mediapipe` runtime loads WASM from a CDN. It is aliased to a stub in `vite.config.ts` so nothing is fetched or bundled for it.)

**Accept / reject rules** (training photos, `detectAndCropFace(…, 'strict')`):

| Faces found | Result |
| --- | --- |
| 0 | skipped: "no face found" |
| exactly 1 | accepted and cropped |
| more than 1 | skipped: "multiple faces" (it's unclear who the label refers to) |

Faces smaller than **45% of the width of the largest face** don't count as a "second face". That covers a tiny poster, or someone far in the background. A person standing just behind the subject still counts, so the photo is rejected. Upload results are summarised in one toast, for example "Added 24 to Bob. Skipped: 6 multiple faces." Webcam captures skip bad frames silently and report the count when you release the button.

**The crop** (`cropRegion()` / `cropFace()`):

1. Take the face box (x, y, width, height) from BlazeFace.
2. Make it square around the box centre: `side = max(width, height) × (1 + 2 × 0.20)`, so 20% margin on each side, 1.4× in total. The margin keeps hairline, ears and jaw, which help tell people apart.
3. Draw that square into a 224×224 canvas. Where the square goes past the photo's edge, the canvas is mid-gray (128,128,128).
4. Store it as a JPEG (quality 0.92) data URL. That one image is both the thumbnail you see and the model input.

Webcam frames are captured **unmirrored** (true orientation, like uploaded photos), even though the preview is shown mirrored like a mirror. The horizontal-flip augmentation covers both orientations anyway.

**At inference** (`detectAndCropFace(…, 'largest')`), several faces are allowed: the **largest** face is classified and the UI says "2 faces, showing the largest".

---

### Step 2: Frozen MobileNet V2 feature extractor

**What it is.** MobileNet V2 (width 1.0, 224×224 input) is a convolutional network trained on the 1.2-million-photo ImageNet dataset to recognise 1000 object categories. Its last layer is the 1000-way classifier. The layer just before it is a **global average pool producing 1280 numbers**: a compact, general description of what is in the image. That penultimate layer is our feature extractor.

- Loaded with `@tensorflow-models/mobilenet` (`version: 2, alpha: 1.0`) from `public/models/mobilenet/`. This is the same TF-Hub model (`google/imagenet/mobilenet_v2_100_224/classification/2`) the package downloads by default, about 14 MB.
- Features are read from graph node `module_apply_default/MobilenetV2/Logits/AvgPool` via `model.infer(image, true)`, giving a `[n, 1280]` tensor.
- **Frozen:** it is a TF.js *GraphModel*, so its weights are graph constants and cannot be trained. The app only ever runs inference on it.

**Preprocessing: pixels to [-1, 1].** MobileNet V2 expects pixel values in [-1, 1]. This particular TF-Hub graph takes **[0, 1]** input and does the `x × 2 − 1` rescale to [-1, 1] *inside* the graph. The `mobilenet` package knows this only when it downloads the model itself. Because we self-host, the app passes `inputRange: [0, 1]` explicitly; the package then divides raw pixels by 255, and the graph rescales to [-1, 1]. This was verified: an LFW photo of Colin Powell comes out as "Windsor tie 52%, suit 43%", which is correct, while the wrong range produces garbage. `metadata.json` records both `normalization: "[-1,1]"` and the graph's `inputRange: [0, 1]`.

---

### Step 3: Data augmentation (6× per photo)

Every training photo becomes **6 variants** before feature extraction (`generateAugmentedVariants()`). All of them are made with TF.js ops on the GPU, so there are no canvas filters and they behave the same in every browser:

| # | Variant | How |
| --- | --- | --- |
| 0 | original | as cropped |
| 1 | horizontal flip | `tf.image.flipLeftRight` |
| 2 | brightness ±15% | pixels × 1.15 or × 0.85, clipped to 0–255 |
| 3 | rotation ±10° | `tf.image.rotateWithOffset`, gray fill |
| 4 | zoom out 0.9× | `tf.image.cropAndResize` with a box 1/0.9 the image size, gray padding |
| 5 | zoom in 1.1× | `tf.image.cropAndResize` with a centred box 1/1.1 the image size |

The direction of brightness and rotation (+ or −) is chosen from a hash of the sample id. Half the photos get brighter and half darker, and likewise for rotation. Each photo yields exactly 6 variants, and they are the same on every retrain, so their features can be cached.

**Why it helps.** The head sees slightly different lighting, tilt, scale and mirror image for every photo. It learns that those changes don't change who it is, which gives roughly 6× the training data with no extra uploads. Augmentation is used **only for training**: validation and live predictions use the original crop.

---

### Step 4: The classification head and how it's trained

**Architecture** (`buildHead()`), all layers named so the exports are predictable:

| Layer | Output | Parameters | Purpose |
| --- | --- | --- | --- |
| Input | 1280 | – | MobileNet features of one face |
| `head_dense_1` Dense 256, ReLU, L2(0.001) | 256 | 327,936 | learns combinations of features that identify people |
| `head_bn` BatchNormalization | 256 | 1,024 | keeps activations well-scaled, speeds up and steadies training |
| `head_dropout_1` Dropout 0.4 | 256 | – | randomly drops 40% of units while training, against memorising |
| `head_dense_2` Dense 128, ReLU, L2(0.001) | 128 | 32,896 | second, smaller representation |
| `head_dropout_2` Dropout 0.3 | 128 | – | more regularisation |
| `head_output` Dense N, softmax | N | 129 × N | one probability per class, summing to 1 |

That's about 362k parameters for 3 classes, 1.4 MB as float32. **L2 regularisation** adds `0.001 × Σw²` to the loss, which keeps weights small and the decision boundaries smooth. With only 20–80 photos per person, regularisation (L2, batch norm, two dropouts) is what stops the head from simply memorising the training photos.

**Compile:** Adam optimizer (learning rate 0.0005), categorical cross-entropy loss, accuracy metric.

**Train/validation split, by photo** (`stratifiedSplit()`): each class keeps **20% of its photos** (at least 1) as a validation set. The split is made **before augmentation**, by photo. If it were made after augmentation, a flipped or brightened copy of a validation photo would sit in the training set, the model would effectively have seen the test, and the Accuracy Report would be inflated. Validation uses only the **original** crop of each held-out photo, which is what the model will see in real use. (This also avoids a tfjs quirk: `validationSplit` takes the *last* 20% of the arrays, which here would be mostly one class.)

| | Training set | Validation set |
| --- | --- | --- |
| Photos | 80% of each class | 20% of each class |
| Images used | all 6 variants of each photo | original crop only |

**Class weights.** Each class is weighted by `total / (classes × class count)`. A smaller class, typically Unknown, therefore counts more per image and isn't drowned out by people with many photos.

**Fitting.** Up to **40 epochs**, batch size 16, shuffled every epoch. After each epoch the UI gets loss, accuracy, validation loss and validation accuracy, which drive the live charts.

**Early stopping, with best-weights restore.** After each epoch the trainer checks validation loss:

- If it improved (by more than 0.0001), the current weights are snapshotted.
- If it hasn't improved for **5 epochs** (the patience), training stops.
- At the end, the snapshot from the **best epoch** is restored.

TF.js's built-in `earlyStopping` callback can stop but doesn't implement `restoreBestWeights`, so the trainer keeps its own copy. **Stop and keep best** on the training screen uses the same mechanism.

**Feature cache.** Features of all 6 variants are cached per sample id. Retraining after adding photos only sends the new photos through MobileNet. Deleting a photo drops its cache entry.

---

### Step 5: The Accuracy Report

After training, the restored best head predicts the **held-out validation photos** (`evaluate()` in `evaluator.ts`):

- **Confusion matrix:** `matrix[actual][predicted] = count`. Rows are the true person, columns the prediction; the diagonal is correct answers. Each cell is shaded by its share of the row, on a single-hue blue scale, and labelled with the count and percentage. Hover a cell for details such as "Tony Blair photos predicted as Colin Powell: 1 of 6".
- **Precision** for a class = diagonal ÷ column sum. *When the model says "Alice", how often is it really Alice?* Low precision means other people get mistaken for Alice.
- **Recall** for a class = diagonal ÷ row sum. *Of Alice's test photos, how many did it recognise?* Low recall means Alice often gets called someone else.
- **F1** = harmonic mean of precision and recall.
- **Overall accuracy** = diagonal ÷ all test photos.
- **Test / Samples** columns show how many held-out photos each row is based on, and the class's total photo count.

**Warnings:**

- **Weak class:** any class with precision or recall below **70%** gets "Add more photos of X, ideally from different angles/lighting", with the reason (for example "only 33% of Unknown's test photos were recognised").
- **Small test set:** if a class has fewer than 10 held-out photos, the report says how much a single photo moves its recall. With 4 test photos that's 25 points per photo, so treat the numbers as rough.

**Threshold impact.** The report replays the validation predictions through the confidence gate (Step 6) at the current slider value. It shows what share of test photos get a confident answer, how accurate those answers are, and how many photos of known people would become "Not sure".

---

### Step 6: Inference with confidence gating

For every webcam frame (one at a time, never overlapping) or uploaded snapshot (`predictImage()` → `predictWithConfidenceThreshold()`):

```
faces = BlazeFace(image)                       # no face  -> "No face detected"
crop  = cropFace(largest face, 20% margin)     # several -> use the largest, say so
probs = head(MobileNet(crop))                  # softmax over your classes
top   = argmax(probs)

if class[top] is "Unknown"     -> "Unknown"             (the model recognised a stranger)
elif probs[top] < threshold    -> "Unknown / Not sure"  (not confident enough)
else                           -> class[top]            e.g. "Alice (94%)"
```

Softmax always gives *some* class the highest probability, even for a stranger. The threshold (default **0.60**, slider 0.30–0.90) is the safety net: a raised threshold rejects more strangers but also more real matches, and a lower one does the opposite. The bars show every class's probability, with a vertical line marking the threshold. The threshold you pick is saved in the export's `metadata.json`.

---

### Step 7: The "Unknown" class

Add the suggested **Unknown** card and fill it with photos of **strangers and other people**, plus yourself in bad lighting if you like. It becomes a normal output class. Instead of having to choose between Alice and Bob for a stranger, the model can answer "this looks like the other-people pile". Together with the threshold, this is what keeps strangers from being force-matched. Aim for roughly 20% of your photos to be Unknown, from many different people. The name "Unknown" is reserved, so a person card can't use it.

---

### Why transfer learning works here (and where it doesn't)

MobileNet already knows a great deal about faces from ImageNet: skin tone, hair, glasses, face shape, beards. A small head can learn which combinations separate *your* few people from a few dozen photos.

But MobileNet was **never trained to tell identities apart**. FaceNet and ArcFace were trained on millions of faces specifically so that two photos of the same person land close together regardless of lighting and pose. A frozen ImageNet model is much weaker at that, so this app:

- works best with **few, visually distinct people** and **training photos that look like the test conditions**;
- **cannot recognise anyone it wasn't trained on**: there is no "enrol from one photo" as with embeddings;
- can be confused by **lighting, pose and camera changes** between training and use.

### Performance and memory

- Feature extraction runs **4 photos (24 augmented images) per MobileNet call** inside `tf.tidy()`. Every intermediate tensor is freed, and the output is downloaded and disposed. Progress is reported per augmented image.
- After training, the only new tensors are the head's 10 weights and Adam's 18 state tensors, which was verified with `tf.memory()`. Predictions and `loadModel()` round-trips leak nothing.
- Training yields to the browser between batches and epochs. It uses `requestAnimationFrame` capped at 50 ms when the tab is visible, and a `MessageChannel` when hidden, so it keeps running in a background tab.
- **Limits:** up to 8 people plus Unknown, and 500 photos per class.
- **Backend:** WebGL is tried first; if it fails, the app falls back to CPU with a warning toast. CPU works, but slowly.

---

## Data requirements

| | Minimum (Train unlocks) | "Good" badge | Recommended |
| --- | --- | --- | --- |
| Each person | 20 photos | 40 | 40–80 |
| Unknown (optional) | 10 photos | 20 | ~20% of all photos |

The **Data Quality Tips** panel on the Collect screen:

- vary **angles** (front, 45°, profile);
- vary **lighting** (indoor, outdoor, backlit);
- vary **expressions** (neutral, smiling);
- vary **distance** (close, medium);
- include **about 20% Unknown** photos from many different people.

20 varied photos beat 200 near-identical selfies. If you'll use the webcam at your desk, include webcam photos from your desk.

---

## Export formats

Only the trained **head** and its metadata are exported. BlazeFace and MobileNet are identical for every model and are reloaded at load time.

### TensorFlow.js (`person-model-tfjs.zip`)

| File | Contents |
| --- | --- |
| `model.json` | head architecture (Layers format) + weights manifest |
| `weights.bin` | head weights (~1.4 MB) |
| `metadata.json` | labels, preprocessing and face-crop contract, threshold |
| `convert_to_keras.py` | the Keras script below, for convenience |

```json
{
  "labels": ["Colin Powell", "Tony Blair", "Unknown"],
  "imageSize": 224,
  "normalization": "[-1,1]",
  "baseModel": "mobilenet_v2",
  "faceCrop": true,
  "confidenceThreshold": 0.6,
  "unknownLabel": "Unknown",
  "faceDetector": "mediapipe_blazeface_short",
  "faceMargin": 0.2,
  "featureDim": 1280,
  "mobilenet": {
    "version": 2,
    "alpha": 1,
    "inputRange": [0, 1],
    "embeddingLayer": "module_apply_default/MobilenetV2/Logits/AvgPool",
    "source": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
  },
  "createdAt": "…",
  "generator": "person-recognizer",
  "version": 1
}
```

**Loading it again.** `loadModel()` in `src/ml/predictor.ts` accepts the `.zip` File, the individual files, or a hosted folder:

```ts
import { loadModel } from './ml/predictor';

const model = await loadModel(zipFile);                         // or { modelUrl: '/my-model/model.json' }
const result = await model.predict(imageFile);                  // File/Blob, <img>, <video> or <canvas>
// { person: 'Alice', confidence: 0.94, allProbs: { Alice: 0.94, Bob: 0.03, Unknown: 0.03 }, faceCount: 1, box }
// person = 'Unknown / Not sure' below the threshold (metadata.confidenceThreshold, or pass one),
// person = null when no face was found.
model.dispose();
```

`loadModel()` loads BlazeFace and MobileNet, loads the head and metadata, checks that the head's input size matches MobileNet's 1280 features and its output size matches the labels, and returns `predict()`. `predict()` runs the whole pipeline: detect → crop → features → head → gate. The start screen's **Open an exported model** uses the same function.

### Keras `.h5`

The browser can't write HDF5. The Export dialog's **Keras** tab therefore shows a copyable Python script (also included in the zip as `convert_to_keras.py`). It reads `model.json` + `weights.bin` with numpy, rebuilds the **same head** in `tf.keras` with the same layer names, and saves `person_head.h5` + `labels.txt`:

```bash
unzip person-model-tfjs.zip -d person-model && cd person-model
pip install "tensorflow>=2.12"
python convert_to_keras.py
```

**The MobileNet base must be loaded separately.** The `.h5` is only the head, and its input is the 1280-d feature vector of a face crop, not an image. To classify a photo in Python, reproduce the browser pipeline:

1. detect the face (MediaPipe face detection);
2. crop a square with a 20% margin per side and resize to 224×224;
3. get features from MobileNet V2 1.0 224 (TF-Hub `google/imagenet/mobilenet_v2_100_224`), with pixels scaled to [0, 1];
4. run `head.predict(features)`.

---

## Project structure

```
person-recognizer/
├── index.html · package.json · vite.config.ts · tsconfig.json · tailwind.config.js · postcss.config.js
├── public/models/
│   ├── blazeface/            # BlazeFace short-range (tfjs graph, 0.3 MB)
│   └── mobilenet/            # MobileNet V2 1.0 224 (tfjs graph, 13.4 MB, 4 shards)
├── scripts/fetch-models.mjs  # re-downloads both from TF Hub
└── src/
    ├── config.ts             # limits, thresholds, training defaults, class colors
    ├── types/index.ts        # Person, Sample, Evaluation, TrainedModel, ModelMetadata, Prediction
    ├── store/
    │   ├── useProjectStore.ts  # Zustand: classes, photos, progress, trained model, threshold, screen
    │   └── useToastStore.ts
    ├── ml/
    │   ├── faceDetector.ts   # loadFaceDetector(), detectFaces(), cropFace(), detectAndCropFace()
    │   ├── mobilenet.ts      # initBackend(), loadMobilenet(), extractFeatures()
    │   ├── augmentation.ts   # generateAugmentedVariants() (6 variants)
    │   ├── trainer.ts        # getFeatures() + cache, buildHead(), stratifiedSplit(), trainHead() (early stopping)
    │   ├── evaluator.ts      # confusionMatrix(), computePerClassMetrics(), evaluate(), thresholdStats(), weakClasses()
    │   ├── predictor.ts      # classifyCrop(), predictWithConfidenceThreshold(), predictImage(), loadModel()
    │   ├── exporter.ts       # exportTfjs(), buildMetadata(), kerasPythonSnippet()
    │   └── mediapipeStub.ts  # stands in for the unused MediaPipe runtime
    ├── utils/                # image.ts (decode/snapshot), camera.ts (shared webcam stream), download.ts
    ├── hooks/                # useFaceDetector.ts (loads both models), useTraining.ts, useInference.ts
    └── components/
        ├── Landing, AppHeader, Workspace, PersonCard, SampleGallery, WebcamCapture, FaceOverlay,
        ├── DataQualityTips, TrainingProgress, AccuracyReport, InferencePanel, ExportModal
        └── ui/               # Button, Card, Modal, Tabs, ProgressBar, CodeBlock, Toaster, LineChart, icons
```

---

## Demo test results

The spec's demo test was run in the browser with photos from **Labeled Faces in the Wild** (LFW, a public research dataset). The photos were used locally for testing only and are not part of this repo. Input: 30 photos each of Colin Powell and Tony Blair, plus strangers for Unknown.

| Step | Result |
| --- | --- |
| Face filter | Powell 28/30 and Blair 24/30 accepted. 11 of the 70 photos were rejected as "multiple faces"; these were real background people in news photos. Unknown had 7 of 10, so Train correctly stayed locked; topped up with 6 more strangers to 13. |
| Feature extraction | 65 photos × 6 = 390 images, progress shown 24 at a time |
| Training | early stopping after 16 epochs; weights restored from epoch 11 |
| **Accuracy Report** (14 held-out photos) | **Overall 78.6%**. Colin Powell: precision 71.4%, recall 83.3%. Tony Blair: 83.3% / 100%. Unknown: 100% / 33.3%, with the weak-class warning shown, plus the small-test-set caveat. |
| Snapshot test, 15 never-seen photos | Blair 5/5 correct. Powell 4/5 correct and 1 "Not sure" (52% < 0.60). Strangers 5/5 rejected (4 via Unknown, 1 via the threshold). **No false matches.** |
| Export round-trip | `loadModel(zip)` reproduces in-app probabilities exactly; no leaked tensors |
| Keras | `convert_to_keras.py` on the real export builds `person_head.h5`, which matches the browser head's output to within 1e-7 on the same features |

Small validation sets make per-class numbers coarse: one photo is 17–33 points here. Results vary between runs because the split is random.

---

## Error handling

| Situation | Behaviour |
| --- | --- |
| No face / multiple faces in a photo | skipped; one summary toast per upload, or per webcam hold |
| Camera denied, missing or busy | toast + message in the webcam box with **Try again**; the preview offers "Use a photo instead" |
| Insecure origin / no `getUserMedia` | explains that the camera needs localhost/HTTPS |
| Model files missing | toast naming the URL and `npm run fetch-models` |
| WebGL unavailable | CPU fallback, with a warning toast |
| Out of memory while training | toast suggesting fewer photos; the previous model stays usable |
| Requirements not met | Train disabled, with the exact reason ("Unknown 7/10", duplicate names, reserved name "Unknown") |
| Over 500 photos in a class | extras dropped / capture stops, with a toast |
| Reload with unsaved work | browser "leave page?" prompt (photos live only in memory) |

---

## Limitations

- **Not FaceNet/ArcFace.** It's a frozen-feature + softmax classifier. It recognises the people you train it on and won't generalise to "any person"; that needs a face-embedding model.
- **Diversity matters more than volume.** 20 good, varied photos beat 200 selfies.
- **Lighting and pose mismatch** between training and testing hurts accuracy the most.
- **WebGL recommended**; the CPU fallback is slow.
- **Not for security or authentication.** It is easy to fool (a photo of a photo works) and has no liveness check.
- Nothing is saved: photos live in memory until you export.
- The face detector is short-range: very small or distant faces may not be found.
- Faces at a strong profile angle are detected less reliably than frontal ones.

---

## Design decisions and deviations from the spec

- **Validation split by photo, before augmentation.** Specified as `validationSplit: 0.2`. Implemented as a stratified 80/20 split per class, made before augmentation, with validation on original crops only. Splitting augmented variants would leak near-copies of test photos into training and inflate the report.
- **Early stopping restores the best epoch.** TF.js has no `restoreBestWeights`, so the trainer snapshots the weights itself.
- **Augmentation directions.** "Brightness ±15%" and "rotation ±10°" each produce one variant whose sign alternates per photo. This keeps exactly 6 variants per photo, as specified.
- **Class weights** (balanced) were added so a small Unknown class still counts.
- **Small background faces** (under 45% of the main face's width) are ignored by the multiple-faces rule. Larger second faces are rejected as specified.
- **Max 8 people plus Unknown**, so every class gets its own colour from a colour-blind-safe palette.
- **Models are stored in the repo** instead of downloaded on first run, so the app works offline straight after `npm install`.
- **`inputRange: [0, 1]`** for the self-hosted TF-Hub MobileNet; see Step 2.
