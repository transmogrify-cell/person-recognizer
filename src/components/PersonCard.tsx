import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { MAX_SAMPLES_PER_CLASS } from '../config';
import { detectAndCropFace } from '../ml/faceDetector';
import { classColor, isBusy, sampleQuality, useProjectStore, type Quality } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { canvasToSample, closeImage, decodeFile } from '../utils/image';
import type { Person, Sample } from '../types';
import { SampleGallery } from './SampleGallery';
import { WebcamCapture } from './WebcamCapture';
import { Card } from './ui/Card';
import { AlertIcon, CameraIcon, CheckIcon, HelpIcon, PencilIcon, SpinnerIcon, TrashIcon, UploadIcon } from './ui/icons';

/**
 * Uploads go through the same face pipeline as the webcam: decode -> BlazeFace
 * -> exactly one face -> 224x224 crop with 20% margin. Others are skipped and
 * summarised in one toast.
 */
async function importPhotos(cls: Person, files: File[], onProgress: (done: number) => void): Promise<void> {
  const { addSamples } = useProjectStore.getState();
  const images = files.filter((f) => f.type.startsWith('image/'));
  const notImages = files.length - images.length;
  const room = MAX_SAMPLES_PER_CLASS - cls.samples.length;
  const queue = images.slice(0, Math.max(0, room));
  const tally = { added: 0, none: 0, multiple: 0, unreadable: 0 };
  let pending: Omit<Sample, 'id'>[] = [];

  for (let i = 0; i < queue.length; i++) {
    try {
      const image = await decodeFile(queue[i]);
      try {
        const face = await detectAndCropFace(image, 'strict');
        if (face.status === 'ok') pending.push({ dataUrl: canvasToSample(face.crop), source: 'upload' });
        else tally[face.status]++;
      } finally {
        closeImage(image);
      }
    } catch {
      tally.unreadable++;
    }
    // Add in small chunks so the gallery fills while the rest are processed.
    if (pending.length >= 6 || i === queue.length - 1) {
      tally.added += addSamples(cls.id, pending);
      pending = [];
    }
    onProgress(i + 1);
  }

  const skipped = [
    tally.none && `${tally.none} no face found`,
    tally.multiple && `${tally.multiple} multiple faces`,
    tally.unreadable && `${tally.unreadable} unreadable`,
    notImages && `${notImages} not images`,
    images.length > queue.length && `${images.length - queue.length} over the ${MAX_SAMPLES_PER_CLASS} limit`,
  ].filter(Boolean);
  if (skipped.length) toast.warning(`Added ${tally.added} to ${cls.name}. Skipped: ${skipped.join(', ')}.`);
  else if (tally.added) toast.success(`Added ${tally.added} face${tally.added === 1 ? '' : 's'} to ${cls.name}.`);
}

const BADGES: Record<Quality, { label: string; cls: string; icon: JSX.Element }> = {
  good: { label: 'Good', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-200', icon: <CheckIcon size={12} /> },
  'needs-more': { label: 'Needs more', cls: 'bg-amber-50 text-amber-800 ring-amber-200', icon: <AlertIcon size={12} /> },
  'too-few': { label: 'Too few', cls: 'bg-red-50 text-red-700 ring-red-200', icon: <AlertIcon size={12} /> },
};

export function QualityBadge({ cls }: { cls: Person }) {
  const { quality, min, good } = sampleQuality(cls);
  const b = BADGES[quality];
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${b.cls}`}
      title={`Minimum ${min} to train, ${good}+ recommended`}
    >
      {b.icon}
      {b.label}
    </span>
  );
}

export function PersonCard({ cls, canDelete }: { cls: Person; canDelete: boolean }) {
  const busy = useProjectStore((s) => isBusy(s.progress));
  const webcamOpen = useProjectStore((s) => s.activeWebcamClassId === cls.id);
  const detectorReady = useProjectStore((s) => s.detector.status === 'ready');
  const color = useProjectStore((s) => classColor(s.classes, cls.id));
  const { setActiveWebcam, renameClass, removeClass, removeSample, clearSamples } = useProjectStore.getState();
  const [draft, setDraft] = useState(cls.name);
  const [importing, setImporting] = useState<{ done: number; total: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const isUnknown = cls.kind === 'unknown';
  const locked = busy || importing !== null;
  const { min } = sampleQuality(cls);
  const n = cls.samples.length;

  useEffect(() => setDraft(cls.name), [cls.name]);

  const commitName = () => {
    const name = draft.trim();
    if (!name) setDraft(cls.name);
    else if (name !== cls.name) renameClass(cls.id, name);
  };

  const runImport = async (list: FileList | File[]) => {
    if (!detectorReady) {
      toast.warning('The face detector is still loading.');
      return;
    }
    const files = Array.from(list);
    setImporting({ done: 0, total: files.length });
    try {
      await importPhotos(useProjectStore.getState().classes.find((c) => c.id === cls.id) ?? cls, files, (done) =>
        setImporting({ done, total: files.length }),
      );
    } finally {
      setImporting(null);
    }
  };

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void runImport(e.target.files);
    e.target.value = '';
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (!locked && e.dataTransfer.files.length) void runImport(e.dataTransfer.files);
  };
  const onDelete = () => {
    if (n > 0 && !window.confirm(`Delete "${cls.name}" and its ${n} photos?`)) return;
    removeClass(cls.id);
  };

  return (
    <Card
      className={`flex flex-col ${isUnknown ? 'bg-slate-50' : ''} ${dragOver ? 'ring-2 ring-brand-500' : ''}`}
      onDragOver={(e) => {
        if (locked) return;
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      data-class-card={cls.name}
    >
      <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3">
        <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: color }} aria-hidden="true" />
        {isUnknown ? (
          <div className="flex min-w-0 flex-1 items-center gap-1.5">
            <span className="font-semibold text-slate-900">{cls.name}</span>
            <span
              className="text-slate-400"
              title="Add photos of strangers/other people so the model can say 'I don't know'."
              aria-label="About the Unknown class"
            >
              <HelpIcon size={15} />
            </span>
          </div>
        ) : (
          <div className="flex min-w-0 flex-1 items-center gap-1.5">
            <input
              ref={nameInput}
              value={draft}
              disabled={busy}
              maxLength={40}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitName}
              onKeyDown={(e) => {
                if (e.key === 'Enter') e.currentTarget.blur();
                if (e.key === 'Escape') {
                  setDraft(cls.name);
                  requestAnimationFrame(() => nameInput.current?.blur());
                }
              }}
              aria-label="Person name"
              className="min-w-0 flex-1 rounded-md bg-transparent px-1 py-0.5 font-semibold text-slate-900 outline-none ring-brand-500 hover:bg-slate-50 focus:bg-white focus:ring-2"
            />
            <button
              type="button"
              onClick={() => nameInput.current?.select()}
              disabled={busy}
              className="rounded p-1 text-slate-400 hover:text-slate-700 disabled:hidden"
              aria-label="Rename"
            >
              <PencilIcon size={14} />
            </button>
          </div>
        )}
        <QualityBadge cls={cls} />
        <span className="shrink-0 text-xs font-medium tabular-nums text-slate-500" title={`Minimum ${min} to train`}>
          {n}
        </span>
        <div className="flex shrink-0">
          {n > 0 && (
            <button
              type="button"
              onClick={() => window.confirm(`Remove all ${n} photos from "${cls.name}"?`) && clearSamples(cls.id)}
              disabled={locked}
              className="rounded px-1.5 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 disabled:opacity-40"
            >
              Clear
            </button>
          )}
          {canDelete && (
            <button
              type="button"
              onClick={onDelete}
              disabled={locked}
              className="rounded p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-600 disabled:opacity-40"
              aria-label={`Delete ${cls.name}`}
            >
              <TrashIcon size={16} />
            </button>
          )}
        </div>
      </div>

      {isUnknown && (
        <p className="px-4 pt-3 text-xs leading-relaxed text-slate-500">
          Strangers, other people, and yourself in bad lighting. Teaches the model to answer "I don't know" instead of force-matching. Aim for
          about 20% of your photos.
        </p>
      )}

      <div className="flex flex-1 flex-col gap-3 p-4 sm:flex-row">
        {webcamOpen ? (
          <div className="w-full shrink-0 sm:w-56">
            <WebcamCapture classId={cls.id} disabled={busy} onClose={() => setActiveWebcam(null)} />
          </div>
        ) : (
          <div className="flex shrink-0 gap-2 sm:flex-col">
            <SourceButton icon={<CameraIcon />} label="Webcam" disabled={locked || !detectorReady} onClick={() => setActiveWebcam(cls.id)} />
            <SourceButton
              icon={importing ? <SpinnerIcon /> : <UploadIcon />}
              label={importing ? `${importing.done}/${importing.total}` : 'Upload photos'}
              disabled={locked || !detectorReady}
              onClick={() => fileInput.current?.click()}
            />
            <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={onFileChange} data-upload-for={cls.name} />
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-xs font-medium text-slate-500">
            {n === 0 ? 'Face crops appear here' : `${n} face${n === 1 ? '' : 's'}`}
            {n > 0 && n < min && <span className="text-red-600"> · {min - n} more to train</span>}
          </p>
          <SampleGallery
            samples={cls.samples}
            onDelete={(id) => removeSample(cls.id, id)}
            disabled={locked}
            emptyText={isUnknown ? 'Photos of other people go here.' : 'Hold to capture, upload, or drop photos here. Only the face is kept.'}
          />
        </div>
      </div>
    </Card>
  );
}

function SourceButton({ icon, label, onClick, disabled }: { icon: JSX.Element; label: string; onClick(): void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex h-16 w-24 flex-col items-center justify-center gap-1 rounded-xl bg-brand-50 text-xs font-medium text-brand-700 transition-colors hover:bg-brand-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
    >
      {icon}
      {label}
    </button>
  );
}
