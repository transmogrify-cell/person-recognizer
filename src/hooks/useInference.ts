import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { predictImage, type ImagePrediction } from '../ml/predictor';
import { isBusy, resolveLabels, useProjectStore } from '../store/useProjectStore';
import { closeImage, decodeFile, snapshotVideo } from '../utils/image';

/**
 * Live webcam classification: snapshot a frame, detect + crop the largest face,
 * classify, gate by threshold. One frame at a time (never overlapping), paused
 * while a training run is swapping the head.
 */
export function useLiveInference(video: HTMLVideoElement | null, active: boolean) {
  const [result, setResult] = useState<ImagePrediction | null>(null);
  const [fps, setFps] = useState(0);
  const canvas = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!active || !video) {
      setResult(null);
      return;
    }
    let stopped = false;
    let raf = 0;
    let frames = 0;
    let since = performance.now();
    const tick = async () => {
      if (stopped) return;
      const s = useProjectStore.getState();
      if (s.trained && !isBusy(s.progress) && video.readyState >= 2 && video.videoWidth) {
        try {
          const frame = snapshotVideo(video, (canvas.current ??= document.createElement('canvas')));
          const r = await predictImage(frame, {
            head: s.trained.head,
            labels: resolveLabels(s.trained, s.classes),
            kinds: s.trained.kinds,
            threshold: s.threshold,
          });
          if (!stopped) setResult(r);
          frames++;
          const now = performance.now();
          if (now - since > 1000) {
            setFps(Math.round((frames * 1000) / (now - since)));
            frames = 0;
            since = now;
          }
        } catch (err) {
          console.warn('[inference] frame failed', err);
        }
      }
      if (!stopped) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, [active, video]);

  return { result, fps };
}

/** "Snapshot test": classify one uploaded photo; re-runs when the threshold or model changes. */
export function useSnapshotInference() {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const threshold = useProjectStore((s) => s.threshold);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [result, setResult] = useState<ImagePrediction | null>(null);
  const [running, setRunning] = useState(false);
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);

  useEffect(() => {
    if (!file) return;
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  useEffect(() => {
    if (!file || !trained) return;
    let cancelled = false;
    setRunning(true);
    (async () => {
      const image = await decodeFile(file);
      try {
        const r = await predictImage(image, { head: trained.head, labels, kinds: trained.kinds, threshold });
        if (!cancelled) {
          setSize({ width: image.width, height: image.height });
          setResult(r);
        }
      } finally {
        closeImage(image);
        if (!cancelled) setRunning(false);
      }
    })().catch((err) => {
      console.error(err);
      if (!cancelled) setRunning(false);
    });
    return () => {
      cancelled = true;
    };
  }, [file, trained, labels, threshold]);

  const choose = useCallback((f: File) => {
    setResult(null);
    setFile(f);
  }, []);

  return { url, size, result, running, choose, clear: () => (setFile(null), setUrl(null), setResult(null)) };
}
