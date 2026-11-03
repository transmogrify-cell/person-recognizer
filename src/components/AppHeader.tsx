import { isBusy, modelsReady, useProjectStore, type Screen } from '../store/useProjectStore';
import { SpinnerIcon } from './ui/icons';

export function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#2547eb" />
      <circle cx="16" cy="12.5" r="5" fill="none" stroke="white" strokeWidth="2.5" />
      <path d="M7.5 25c1.5-4 4.7-6 8.5-6s7 2 8.5 6" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

const STEPS: { id: Screen; label: string }[] = [
  { id: 'workspace', label: 'Collect' },
  { id: 'training', label: 'Train' },
  { id: 'report', label: 'Report' },
  { id: 'inference', label: 'Preview' },
];

export function ModelStatus() {
  const detector = useProjectStore((s) => s.detector);
  const mobilenet = useProjectStore((s) => s.mobilenet);
  const backend = useProjectStore((s) => s.backend);
  const ready = useProjectStore(modelsReady);
  if (detector.status === 'error' || mobilenet.status === 'error') {
    return <span className="text-red-600">Model load failed: {detector.error ?? mobilenet.error}</span>;
  }
  if (!ready) {
    return (
      <span className="flex items-center gap-1.5 text-slate-500">
        <SpinnerIcon size={14} />
        Loading {detector.status !== 'ready' ? 'face detector' : 'MobileNet'}…
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-slate-500">
      <span className="h-2 w-2 rounded-full bg-emerald-500" /> BlazeFace + MobileNet V2 · {backend === 'webgl' ? 'GPU' : 'CPU'}
    </span>
  );
}

/** Sticky header with the Collect → Train → Report → Preview steps. */
export function AppHeader() {
  const screen = useProjectStore((s) => s.screen);
  const goTo = useProjectStore((s) => s.goTo);
  const trained = useProjectStore((s) => s.trained);
  const progress = useProjectStore((s) => s.progress);
  const busy = isBusy(progress);
  const hasWork = useProjectStore((s) => s.classes.some((c) => c.samples.length > 0) || s.trained !== null);
  const newProject = useProjectStore((s) => s.newProject);

  const enabled = (id: Screen) => {
    if (busy) return id === 'training';
    if (id === 'workspace') return true;
    if (id === 'training') return progress.phase !== 'idle';
    if (id === 'report') return trained?.evaluation != null;
    return trained !== null;
  };

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <button type="button" onClick={() => !busy && goTo('landing')} className="flex items-center gap-2" aria-label="Start screen">
          <Logo />
          <span className="hidden font-semibold text-slate-900 sm:inline">Person Recognizer</span>
        </button>
        <nav className="ml-2 flex items-center gap-0.5 overflow-x-auto" aria-label="Steps">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              type="button"
              disabled={!enabled(s.id)}
              onClick={() => goTo(s.id)}
              aria-current={screen === s.id ? 'step' : undefined}
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-lg px-2.5 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-300 ${
                screen === s.id ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              <span className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] ${screen === s.id ? 'bg-brand-600 text-white' : 'bg-slate-200 text-slate-600'}`}>
                {i + 1}
              </span>
              {s.label}
            </button>
          ))}
        </nav>
        <div className="ml-auto hidden text-xs lg:block">
          <ModelStatus />
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={() => (!hasWork || window.confirm('Start over? All photos and the trained model will be discarded.')) && newProject()}
          className="whitespace-nowrap rounded-lg px-2.5 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-100 disabled:text-slate-300"
        >
          New project
        </button>
      </div>
    </header>
  );
}
