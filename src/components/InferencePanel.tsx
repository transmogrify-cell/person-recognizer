import { useMemo, useRef, useState, type ChangeEvent } from 'react';
import { THRESHOLD_MAX, THRESHOLD_MIN } from '../config';
import { useLiveInference, useSnapshotInference } from '../hooks/useInference';
import type { ImagePrediction } from '../ml/predictor';
import { classColor, resolveLabels, useProjectStore } from '../store/useProjectStore';
import { useCamera } from '../utils/camera';
import { ExportModal } from './ExportModal';
import { FaceOverlay } from './FaceOverlay';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { CameraIcon, DownloadIcon, ImageIcon, ShieldIcon, SpinnerIcon } from './ui/icons';

type Mode = 'webcam' | 'snapshot';

/** Screen 5: live webcam / snapshot prediction with the confidence gate. */
export function InferencePanel() {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const threshold = useProjectStore((s) => s.threshold);
  const setThreshold = useProjectStore((s) => s.setThreshold);
  const stale = useProjectStore((s) => s.stale);
  const [mode, setMode] = useState<Mode>('webcam');
  const [camOn, setCamOn] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const camera = useCamera(mode === 'webcam' && camOn && trained !== null);
  const live = useLiveInference(camera.videoRef.current, mode === 'webcam' && camOn && camera.status === 'ready');
  const snap = useSnapshotInference();
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);

  if (!trained) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-10">
        <Card className="p-6 text-sm text-slate-600">Train a model first.</Card>
      </main>
    );
  }

  const result = mode === 'webcam' ? live.result : snap.result;
  const video = camera.videoRef.current;
  const pred = result?.prediction;
  const tone = !pred ? 'warn' : pred.isUnknown ? 'unknown' : 'ok';

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = '';
    if (f) snap.choose(f);
  };

  return (
    <main className="mx-auto grid max-w-7xl gap-4 px-4 py-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <Card className="flex flex-col gap-3 p-4">
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 text-sm font-medium" role="radiogroup" aria-label="Input">
          {(['webcam', 'snapshot'] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              onClick={() => setMode(m)}
              className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 ${mode === m ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
            >
              {m === 'webcam' ? <CameraIcon size={16} /> : <ImageIcon size={16} />}
              {m === 'webcam' ? 'Live webcam' : 'Snapshot test'}
            </button>
          ))}
        </div>

        {mode === 'webcam' ? (
          <div
            className="relative overflow-hidden rounded-xl bg-slate-900"
            style={{ aspectRatio: video?.videoWidth ? `${video.videoWidth} / ${video.videoHeight}` : '4 / 3' }}
          >
            {camOn && <video ref={camera.videoRef} muted playsInline autoPlay className="h-full w-full -scale-x-100 object-fill" />}
            {camOn && result?.box && video?.videoWidth && (
              <FaceOverlay boxes={[result.box]} width={video.videoWidth} height={video.videoHeight} mirror tone={tone} label={pred && overlayLabel(pred)} />
            )}
            {(!camOn || camera.error || camera.status === 'starting') && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-sm text-slate-300">
                {camera.status === 'starting' ? (
                  <span className="flex items-center gap-2">
                    <SpinnerIcon /> Starting camera…
                  </span>
                ) : camera.error ? (
                  <>
                    <p>{camera.error}</p>
                    <Button size="sm" onClick={() => setMode('snapshot')}>
                      Use a photo instead
                    </Button>
                  </>
                ) : (
                  <Button variant="primary" onClick={() => setCamOn(true)}>
                    <CameraIcon size={16} /> Start webcam
                  </Button>
                )}
              </div>
            )}
            {camOn && camera.status === 'ready' && (
              <div className="absolute right-2 top-2 flex items-center gap-2">
                <span className="rounded-full bg-slate-900/60 px-2 py-0.5 font-mono text-[11px] text-white">{live.fps} fps</span>
                <button type="button" onClick={() => setCamOn(false)} className="rounded-full bg-slate-900/60 px-2.5 py-1 text-xs font-medium text-white hover:bg-slate-900">
                  Stop
                </button>
              </div>
            )}
          </div>
        ) : (
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="group relative flex min-h-[16rem] items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 text-sm text-slate-500 hover:border-brand-400"
            style={snap.size ? { aspectRatio: `${snap.size.width} / ${snap.size.height}` } : undefined}
          >
            {snap.url ? (
              <>
                <img src={snap.url} alt="Photo being tested" className="h-full w-full object-fill" />
                {snap.result?.box && snap.size && (
                  <FaceOverlay boxes={[snap.result.box]} width={snap.size.width} height={snap.size.height} tone={tone} label={pred && overlayLabel(pred)} />
                )}
                {snap.running && (
                  <span className="absolute inset-0 flex items-center justify-center bg-white/40">
                    <SpinnerIcon />
                  </span>
                )}
                <span className="absolute bottom-2 right-2 rounded-full bg-slate-900/70 px-2.5 py-1 text-xs font-medium text-white opacity-0 group-hover:opacity-100">
                  Choose another photo
                </span>
              </>
            ) : (
              <span className="flex flex-col items-center gap-2">
                <ImageIcon size={28} /> Upload a photo to test
              </span>
            )}
          </button>
        )}
        <input ref={fileInput} type="file" accept="image/*" hidden onChange={onFile} data-testid="snapshot-input" />
        <p className="flex items-start gap-2 text-xs text-slate-500">
          <ShieldIcon size={14} className="mt-0.5 shrink-0" />A learning/demo tool, not for security or authentication. It only knows the people you
          trained it on.
        </p>
      </Card>

      <div className="flex flex-col gap-4">
        <Card className="flex flex-col gap-4 p-5">
          {stale && <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">Photos changed after training; this is the previous model.</p>}
          <ResultHeadline result={result} waiting={mode === 'webcam' ? camOn : snap.url !== null} />
          <ul className="flex flex-col gap-2" data-testid="prob-bars">
            {labels.map((label, i) => {
              const p = pred ? pred.allProbs[label] ?? 0 : 0;
              const top = pred?.topIndex === i;
              return (
                <li key={trained.classIds[i]} className="grid grid-cols-[minmax(0,7rem)_1fr_3rem] items-center gap-3 text-sm">
                  <span className={`truncate ${top ? 'font-semibold text-slate-900' : 'text-slate-600'}`} title={label}>
                    {label}
                  </span>
                  <div className="relative h-6 overflow-hidden rounded-md bg-slate-100">
                    <div
                      className="h-full rounded-md transition-[width] duration-150"
                      style={{ width: `${p * 100}%`, background: classColor(classes, trained.classIds[i]), opacity: top ? 1 : 0.5 }}
                    />
                    <div className="absolute inset-y-0 w-px bg-slate-900/50" style={{ left: `${threshold * 100}%` }} title="Confidence threshold" />
                  </div>
                  <span className="text-right font-mono text-xs tabular-nums text-slate-700">{pred ? `${Math.round(p * 100)}%` : '–'}</span>
                </li>
              );
            })}
          </ul>
          <div className="rounded-xl bg-slate-50 p-3">
            <div className="flex items-center justify-between text-sm">
              <label htmlFor="threshold" className="font-medium text-slate-800">
                Confidence threshold
              </label>
              <span className="font-mono tabular-nums text-slate-700">{threshold.toFixed(2)}</span>
            </div>
            <input
              id="threshold"
              type="range"
              min={THRESHOLD_MIN}
              max={THRESHOLD_MAX}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="mt-2 w-full accent-brand-600"
            />
            <p className="mt-1 text-xs text-slate-500">
              Below {Math.round(threshold * 100)}% confidence the answer becomes "Unknown / Not sure" (the line on each bar), so strangers aren't
              forced into a known person.
            </p>
          </div>
          <Button variant="secondary" onClick={() => setExportOpen(true)}>
            <DownloadIcon size={16} /> Export Model
          </Button>
        </Card>
      </div>
      <ExportModal open={exportOpen} onClose={() => setExportOpen(false)} />
    </main>
  );
}

function overlayLabel(pred: NonNullable<ImagePrediction['prediction']>) {
  return `${pred.person} ${Math.round(pred.confidence * 100)}%`;
}

function ResultHeadline({ result, waiting }: { result: ImagePrediction | null; waiting: boolean }) {
  let big = 'Waiting for a face';
  let sub = waiting ? 'Looking for a face…' : 'Start the webcam or upload a photo.';
  let cls = 'text-slate-400';
  if (result) {
    if (result.face === 'none') {
      big = 'No face detected';
      sub = 'Face the camera, with your face well lit.';
    } else if (result.prediction) {
      const p = result.prediction;
      big = p.person;
      cls = p.isUnknown ? 'text-slate-500' : 'text-slate-900';
      sub =
        p.reason === 'below-threshold'
          ? `Top match was below the threshold (${Math.round(p.confidence * 100)}%).`
          : p.reason === 'unknown-class'
            ? `Matched the Unknown class (${Math.round(p.confidence * 100)}%).`
            : `Top match: ${p.person} (${Math.round(p.confidence * 100)}%)`;
      if (result.faceCount > 1) sub += ` · ${result.faceCount} faces, showing the largest.`;
    }
  }
  return (
    <div aria-live="polite">
      <p className={`truncate text-4xl font-bold tracking-tight ${cls}`} data-testid="predicted-person">
        {big}
      </p>
      <p className="mt-1 text-sm text-slate-500">{sub}</p>
    </div>
  );
}
