import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { CAPTURE_FPS, MAX_SAMPLES_PER_CLASS } from '../config';
import { cropFace, detectFaces, significantFaces } from '../ml/faceDetector';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { useCamera } from '../utils/camera';
import { canvasToSample, snapshotVideo } from '../utils/image';
import type { FaceBox } from '../types';
import { FaceOverlay } from './FaceOverlay';
import { Button } from './ui/Button';
import { SpinnerIcon, XIcon } from './ui/icons';

/**
 * Live webcam with face boxes and "Hold to capture". One loop at CAPTURE_FPS:
 * snapshot the frame, detect faces, draw the boxes, and, while the button is
 * held and exactly one face is visible, crop it and add it as a sample.
 */
export function WebcamCapture({ classId, disabled, onClose }: { classId: string; disabled?: boolean; onClose(): void }) {
  const { videoRef, status, error, retry } = useCamera(true);
  const addSamples = useProjectStore((s) => s.addSamples);
  const count = useProjectStore((s) => s.classes.find((c) => c.id === classId)?.samples.length ?? 0);
  const [recording, setRecording] = useState(false);
  const [faces, setFaces] = useState<{ boxes: FaceBox[]; width: number; height: number } | null>(null);
  const recordingRef = useRef(false);
  const skipped = useRef({ none: 0, multiple: 0, added: 0 });

  const stop = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setRecording(false);
    const { none, multiple, added } = skipped.current;
    const parts = [none && `${none} with no face found`, multiple && `${multiple} with multiple faces`].filter(Boolean);
    if (parts.length) toast.warning(`Captured ${added}. Skipped ${parts.join(' and ')}.`);
  }, []);

  const start = useCallback(() => {
    if (status !== 'ready' || disabled || recordingRef.current) return;
    skipped.current = { none: 0, multiple: 0, added: 0 };
    recordingRef.current = true;
    setRecording(true);
  }, [disabled, status]);

  useEffect(() => {
    if (status !== 'ready') return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const frameCanvas = document.createElement('canvas');
    const loop = async () => {
      if (stopped) return;
      const t0 = performance.now();
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.videoWidth) {
        try {
          const frame = snapshotVideo(video, frameCanvas);
          const boxes = significantFaces(await detectFaces(frame));
          if (stopped) return;
          setFaces({ boxes, width: frame.width, height: frame.height });
          if (recordingRef.current) {
            if (boxes.length === 1) {
              const added = addSamples(classId, [{ dataUrl: canvasToSample(cropFace(frame, boxes[0])), source: 'webcam' }]);
              skipped.current.added += added;
              if (added === 0) {
                toast.info(`This class has reached the ${MAX_SAMPLES_PER_CLASS}-photo limit.`);
                stop();
              }
            } else if (boxes.length === 0) skipped.current.none++;
            else skipped.current.multiple++;
          }
        } catch (err) {
          console.warn('[webcam] detection failed', err);
        }
      }
      timer = setTimeout(loop, Math.max(0, 1000 / CAPTURE_FPS - (performance.now() - t0)));
    };
    void loop();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [status, videoRef, addSamples, classId, stop]);

  useEffect(() => stop, [stop]);
  useEffect(() => {
    if (disabled) stop();
  }, [disabled, stop]);
  useEffect(() => {
    window.addEventListener('blur', stop);
    return () => window.removeEventListener('blur', stop);
  }, [stop]);

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointer: pointerup/leave still stop recording */
    }
    start();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) {
      e.preventDefault();
      start();
    }
  };

  const full = count >= MAX_SAMPLES_PER_CLASS;
  const faceCount = faces?.boxes.length ?? 0;
  const chip =
    status !== 'ready' || !faces
      ? null
      : faceCount === 1
        ? { text: 'Face found', cls: 'bg-emerald-600' }
        : faceCount === 0
          ? { text: 'No face', cls: 'bg-slate-900/70' }
          : { text: `${faceCount} faces`, cls: 'bg-amber-500 text-slate-950' };

  return (
    <div className="flex flex-col gap-2">
      <div
        className="relative overflow-hidden rounded-xl bg-slate-900"
        style={{ aspectRatio: faces ? `${faces.width} / ${faces.height}` : '4 / 3' }}
      >
        <video ref={videoRef} muted playsInline autoPlay className="h-full w-full -scale-x-100 object-fill" />
        {faces && <FaceOverlay boxes={faces.boxes} width={faces.width} height={faces.height} mirror tone={faceCount === 1 ? 'ok' : 'warn'} />}
        {status === 'starting' && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-slate-300">
            <SpinnerIcon /> Starting camera…
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center text-sm text-slate-200">
            <p>{error}</p>
            {status !== 'unsupported' && (
              <Button size="sm" onClick={retry}>
                Try again
              </Button>
            )}
          </div>
        )}
        {chip && (
          <span className={`absolute bottom-2 left-2 rounded-full px-2 py-0.5 text-xs font-semibold text-white ${chip.cls}`}>{chip.text}</span>
        )}
        {recording && (
          <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded-full bg-red-600 px-2 py-0.5 text-xs font-semibold text-white">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" /> REC
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          aria-label="Close webcam"
          className="absolute right-2 top-2 rounded-full bg-slate-900/60 p-1 text-white hover:bg-slate-900"
        >
          <XIcon size={16} />
        </button>
      </div>
      <button
        type="button"
        disabled={disabled || status !== 'ready' || full}
        onPointerDown={onPointerDown}
        onPointerUp={stop}
        onPointerCancel={stop}
        onPointerLeave={stop}
        onLostPointerCapture={stop}
        onKeyDown={onKeyDown}
        onKeyUp={(e) => (e.key === ' ' || e.key === 'Enter') && stop()}
        onBlur={stop}
        onContextMenu={(e) => e.preventDefault()}
        style={{ touchAction: 'none' }}
        className={`h-10 w-full select-none rounded-lg text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 ${
          recording ? 'bg-red-600 text-white' : 'bg-brand-600 text-white hover:bg-brand-700'
        }`}
      >
        {full ? 'Photo limit reached' : recording ? 'Capturing…' : 'Hold to capture'}
      </button>
    </div>
  );
}
