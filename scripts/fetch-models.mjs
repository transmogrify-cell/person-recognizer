#!/usr/bin/env node
// One-time setup: downloads the two pre-trained models into public/models so
// the app runs fully offline afterwards (they are also committed to the repo).
//
//   public/models/blazeface/  MediaPipe BlazeFace short-range face detector (tfjs graph, ~0.4 MB)
//   public/models/mobilenet/  MobileNet V2 1.0 224 ImageNet classifier (tfjs graph, ~14 MB);
//                             its 1280-d global-average-pool layer is our feature extractor
//
// These are exactly the files @tensorflow-models/face-detection and
// @tensorflow-models/mobilenet fetch from TF Hub by default.
//
// Usage: npm run fetch-models   (add --force to re-download)

import { access, mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
const MODELS = {
  blazeface: 'https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1',
  mobilenet: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
};
const force = process.argv.includes('--force');

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function get(url) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`GET ${url}: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

for (const [name, base] of Object.entries(MODELS)) {
  const dir = join(ROOT, name);
  if (!force && (await exists(join(dir, 'model.json')))) {
    console.log(`${name}: already present (use --force to re-download)`);
    continue;
  }
  await mkdir(dir, { recursive: true });
  // TF Hub serves tfjs files when asked with ?tfjs-format=file.
  const modelJson = await get(`${base}/model.json?tfjs-format=file`);
  const manifest = JSON.parse(modelJson.toString('utf8'));
  await writeFile(join(dir, 'model.json'), modelJson);
  let bytes = modelJson.length;
  for (const group of manifest.weightsManifest) {
    for (const path of group.paths) {
      const shard = await get(`${base}/${path}?tfjs-format=file`);
      await writeFile(join(dir, path), shard);
      bytes += shard.length;
    }
  }
  console.log(`${name}: ${(bytes / 1024 / 1024).toFixed(1)} MB -> ${dir}`);
}
