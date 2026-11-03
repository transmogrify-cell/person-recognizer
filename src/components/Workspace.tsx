import { useState } from 'react';
import { DEFAULT_TRAINING, MAX_PERSONS } from '../config';
import { useTraining } from '../hooks/useTraining';
import { VARIANTS_PER_IMAGE } from '../ml/augmentation';
import { useProjectStore } from '../store/useProjectStore';
import { DataQualityTips } from './DataQualityTips';
import { PersonCard } from './PersonCard';
import { Button } from './ui/Button';
import { ChevronIcon, HelpIcon, ScanIcon, SettingsIcon, SpinnerIcon, UserPlusIcon } from './ui/icons';

export function Workspace() {
  const classes = useProjectStore((s) => s.classes);
  const trained = useProjectStore((s) => s.trained);
  const stale = useProjectStore((s) => s.stale);
  const { addPerson, addUnknownClass, goTo } = useProjectStore.getState();
  const { train, busy, readiness } = useTraining();
  const [advanced, setAdvanced] = useState(false);

  const persons = classes.filter((c) => c.kind === 'person');
  const hasUnknown = classes.some((c) => c.kind === 'unknown');
  const photos = classes.reduce((n, c) => n + c.samples.length, 0);

  return (
    <div className="pb-48 sm:pb-36">
      <main className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6">
        <DataQualityTips />
        <div className="grid gap-4 xl:grid-cols-2">
          {classes.map((c) => (
            <PersonCard key={c.id} cls={c} canDelete={c.kind === 'unknown' || persons.length > 1} />
          ))}
          {!hasUnknown && (
            <button
              type="button"
              onClick={addUnknownClass}
              disabled={busy}
              className="flex min-h-[10rem] flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center transition-colors hover:border-brand-400 hover:bg-white disabled:opacity-50"
            >
              <span className="flex items-center gap-1.5 font-semibold text-slate-700">
                <HelpIcon size={18} /> Add the "Unknown" class (recommended)
              </span>
              <span className="max-w-sm text-sm text-slate-500">
                Add photos of strangers/other people so the model can say "I don't know" instead of picking the closest person.
              </span>
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={addPerson}
          disabled={busy || persons.length >= MAX_PERSONS}
          className="flex h-14 items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 text-sm font-semibold text-slate-500 transition-colors hover:border-brand-400 hover:bg-white hover:text-brand-700 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300 disabled:hover:bg-transparent"
        >
          <UserPlusIcon />
          {persons.length >= MAX_PERSONS ? `Maximum of ${MAX_PERSONS} people` : 'Add Person'}
        </button>
      </main>

      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-slate-200 bg-white/95 shadow-[0_-4px_16px_rgba(15,23,42,0.06)] backdrop-blur">
        {advanced && <TrainingSettings disabled={busy} />}
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
          <div className="min-w-0 flex-1 text-sm">
            {readiness.ok ? (
              <p className="text-slate-700">
                <b className="font-semibold">Ready:</b> {classes.length} classes, {photos} photos ({photos * VARIANTS_PER_IMAGE} with augmentation).
                {trained && stale && <span className="text-amber-700"> Photos changed since the last training.</span>}
              </p>
            ) : (
              <p className="text-slate-500">{readiness.reason}</p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="ghost" size="lg" onClick={() => setAdvanced((v) => !v)} aria-expanded={advanced}>
              <SettingsIcon />
              <span className="hidden sm:inline">Advanced</span>
              <ChevronIcon size={16} className={advanced ? '' : 'rotate-180'} />
            </Button>
            {trained && !busy && (
              <Button variant="secondary" size="lg" onClick={() => goTo('inference')}>
                <ScanIcon /> Preview
              </Button>
            )}
            <Button variant="primary" size="lg" onClick={() => void train()} disabled={busy || !readiness.ok} title={readiness.reason}>
              {busy && <SpinnerIcon />}
              {busy ? 'Training…' : trained ? 'Retrain Model' : 'Train Model'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrainingSettings({ disabled }: { disabled?: boolean }) {
  const params = useProjectStore((s) => s.params);
  const setParams = useProjectStore((s) => s.setParams);
  const field =
    'mt-1 h-9 w-full rounded-lg border-0 bg-white px-2.5 text-sm text-slate-900 ring-1 ring-inset ring-slate-300 focus:ring-2 focus:ring-brand-500 disabled:bg-slate-50';
  const clamp = (v: number, lo: number, hi: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo);
  return (
    <fieldset disabled={disabled} className="mx-auto grid max-w-7xl grid-cols-2 gap-3 border-b border-slate-100 px-4 py-4 sm:grid-cols-5">
      <label className="text-xs font-medium text-slate-600">
        Max epochs
        <input type="number" min={1} max={300} value={params.epochs} onChange={(e) => setParams({ epochs: clamp(Math.round(+e.target.value), 1, 300) })} className={field} />
      </label>
      <label className="text-xs font-medium text-slate-600">
        Early-stop patience
        <input type="number" min={1} max={50} value={params.patience} onChange={(e) => setParams({ patience: clamp(Math.round(+e.target.value), 1, 50) })} className={field} />
      </label>
      <label className="text-xs font-medium text-slate-600">
        Batch size
        <select value={params.batchSize} onChange={(e) => setParams({ batchSize: +e.target.value })} className={field}>
          {[8, 16, 32, 64].map((b) => (
            <option key={b}>{b}</option>
          ))}
        </select>
      </label>
      <label className="text-xs font-medium text-slate-600">
        Learning rate
        <input
          type="number"
          step={0.0001}
          min={0.00001}
          max={0.1}
          value={params.learningRate}
          onChange={(e) => +e.target.value > 0 && setParams({ learningRate: clamp(+e.target.value, 0.00001, 0.1) })}
          className={field}
        />
      </label>
      <div className="flex items-end">
        <button type="button" onClick={() => setParams({ ...DEFAULT_TRAINING })} className="h-9 text-xs font-medium text-brand-700 hover:underline">
          Reset defaults
        </button>
      </div>
    </fieldset>
  );
}
