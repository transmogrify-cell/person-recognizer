import { create } from 'zustand';
import {
  DEFAULT_THRESHOLD,
  DEFAULT_TRAINING,
  GOOD_SAMPLES_PERSON,
  GOOD_SAMPLES_UNKNOWN,
  MAX_PERSONS,
  MAX_SAMPLES_PER_CLASS,
  MIN_SAMPLES_PERSON,
  MIN_SAMPLES_UNKNOWN,
  PERSON_COLORS,
  UNKNOWN_COLOR,
  UNKNOWN_LABEL,
} from '../config';
import { clearFeatureCache, forgetFeatures } from '../ml/trainer';
import type { Person, Sample, TrainedModel, TrainingParams, TrainingProgress } from '../types';

export type Screen = 'landing' | 'workspace' | 'training' | 'report' | 'inference';

export interface ModelStatus {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error?: string;
}

interface ProjectStore {
  screen: Screen;
  classes: Person[];
  params: TrainingParams;
  progress: TrainingProgress;
  trained: TrainedModel | null;
  stale: boolean;
  threshold: number;
  activeWebcamClassId: string | null;
  detector: ModelStatus;
  mobilenet: ModelStatus;
  backend?: string;

  goTo(screen: Screen): void;
  newProject(): void;
  addPerson(): void;
  addUnknownClass(): void;
  removeClass(id: string): void;
  renameClass(id: string, name: string): void;
  addSamples(classId: string, samples: Omit<Sample, 'id'>[]): number;
  removeSample(classId: string, sampleId: string): void;
  clearSamples(classId: string): void;
  setParams(params: Partial<TrainingParams>): void;
  setProgress(progress: Partial<TrainingProgress>): void;
  setTrained(trained: TrainedModel | null): void;
  loadTrained(model: Omit<TrainedModel, 'classIds'>, threshold: number): void;
  setThreshold(threshold: number): void;
  setActiveWebcam(classId: string | null): void;
  setModelStatus(which: 'detector' | 'mobilenet', status: ModelStatus): void;
  setBackend(backend: string): void;
}

export const uid = (): string =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

const makePerson = (name: string): Person => ({ id: uid(), name, kind: 'person', samples: [] });

const initialProgress = (): TrainingProgress => ({
  phase: 'idle',
  extracted: 0,
  extractTotal: 0,
  photos: 0,
  epoch: 0,
  epochFraction: 0,
  totalEpochs: DEFAULT_TRAINING.epochs,
  history: [],
});

export const useProjectStore = create<ProjectStore>((set, get) => ({
  screen: 'landing',
  classes: [makePerson('Person 1'), makePerson('Person 2')],
  params: { ...DEFAULT_TRAINING },
  progress: initialProgress(),
  trained: null,
  stale: false,
  threshold: DEFAULT_THRESHOLD,
  activeWebcamClassId: null,
  detector: { status: 'idle' },
  mobilenet: { status: 'idle' },

  goTo: (screen) => set({ screen }),

  newProject() {
    get().trained?.head.dispose();
    clearFeatureCache();
    set({
      classes: [makePerson('Person 1'), makePerson('Person 2')],
      progress: initialProgress(),
      trained: null,
      stale: false,
      activeWebcamClassId: null,
      screen: 'workspace',
    });
  },

  addPerson() {
    const { classes, trained } = get();
    const persons = classes.filter((c) => c.kind === 'person');
    if (persons.length >= MAX_PERSONS) return;
    let n = persons.length + 1;
    while (classes.some((c) => c.name === `Person ${n}`)) n++;
    // Keep "Unknown" last.
    const unknown = classes.filter((c) => c.kind === 'unknown');
    set({ classes: [...persons, makePerson(`Person ${n}`), ...unknown], stale: trained !== null });
  },

  addUnknownClass() {
    const { classes, trained } = get();
    if (classes.some((c) => c.kind === 'unknown')) return;
    set({ classes: [...classes, { id: uid(), name: UNKNOWN_LABEL, kind: 'unknown', samples: [] }], stale: trained !== null });
  },

  removeClass(id) {
    const cls = get().classes.find((c) => c.id === id);
    if (!cls) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.filter((c) => c.id !== id),
      activeWebcamClassId: s.activeWebcamClassId === id ? null : s.activeWebcamClassId,
      stale: s.trained !== null,
    }));
  },

  renameClass(id, name) {
    set((s) => ({ classes: s.classes.map((c) => (c.id === id && c.kind === 'person' ? { ...c, name } : c)) }));
  },

  addSamples(classId, samples) {
    const cls = get().classes.find((c) => c.id === classId);
    if (!cls) return 0;
    const room = Math.max(0, MAX_SAMPLES_PER_CLASS - cls.samples.length);
    const accepted = samples.slice(0, room).map((s) => ({ ...s, id: uid() }));
    if (accepted.length === 0) return 0;
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: [...c.samples, ...accepted] } : c)),
      stale: s.trained !== null,
    }));
    return accepted.length;
  },

  removeSample(classId, sampleId) {
    forgetFeatures([sampleId]);
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: c.samples.filter((x) => x.id !== sampleId) } : c)),
      stale: s.trained !== null,
    }));
  },

  clearSamples(classId) {
    const cls = get().classes.find((c) => c.id === classId);
    if (!cls?.samples.length) return;
    forgetFeatures(cls.samples.map((s) => s.id));
    set((s) => ({
      classes: s.classes.map((c) => (c.id === classId ? { ...c, samples: [] } : c)),
      stale: s.trained !== null,
    }));
  },

  setParams: (params) => set((s) => ({ params: { ...s.params, ...params } })),
  setProgress: (progress) => set((s) => ({ progress: { ...s.progress, ...progress } })),

  setTrained(trained) {
    const previous = get().trained;
    set({ trained, stale: false });
    // Predictions run synchronously inside tf.tidy, so nothing is mid-way through the old head.
    if (previous && previous.head !== trained?.head) previous.head.dispose();
  },

  loadTrained(model, threshold) {
    get().trained?.head.dispose();
    clearFeatureCache();
    const classes: Person[] = model.labels.map((name, i) => ({ id: uid(), name, kind: model.kinds[i], samples: [] }));
    set({
      classes,
      trained: { ...model, classIds: classes.map((c) => c.id) },
      stale: false,
      threshold,
      progress: { ...initialProgress(), phase: 'done' },
      activeWebcamClassId: null,
      screen: 'inference',
    });
  },

  setThreshold: (threshold) => set({ threshold }),
  setActiveWebcam: (activeWebcamClassId) => set({ activeWebcamClassId }),
  setModelStatus: (which, status) => set({ [which]: status } as Pick<ProjectStore, 'detector' | 'mobilenet'>),
  setBackend: (backend) => set({ backend }),
}));

// ───────────────────────────── Selectors ─────────────────────────────

export const isBusy = (p: TrainingProgress): boolean =>
  p.phase === 'extracting' || p.phase === 'training' || p.phase === 'evaluating';

export const modelsReady = (s: { detector: ModelStatus; mobilenet: ModelStatus }) =>
  s.detector.status === 'ready' && s.mobilenet.status === 'ready';

export type Quality = 'good' | 'needs-more' | 'too-few';

export function sampleQuality(cls: Person): { quality: Quality; min: number; good: number } {
  const min = cls.kind === 'unknown' ? MIN_SAMPLES_UNKNOWN : MIN_SAMPLES_PERSON;
  const good = cls.kind === 'unknown' ? GOOD_SAMPLES_UNKNOWN : GOOD_SAMPLES_PERSON;
  const n = cls.samples.length;
  return { quality: n >= good ? 'good' : n >= min ? 'needs-more' : 'too-few', min, good };
}

export interface Readiness {
  ok: boolean;
  reason?: string;
}

export function trainingReadiness(classes: Person[], ready: boolean): Readiness {
  const persons = classes.filter((c) => c.kind === 'person');
  if (persons.length === 0) return { ok: false, reason: 'Add at least one person.' };
  if (classes.length < 2) return { ok: false, reason: 'Add a second person or the Unknown class.' };
  const seen = new Set<string>();
  for (const c of persons) {
    const key = c.name.trim().toLowerCase();
    if (!key) return { ok: false, reason: 'Every person needs a name.' };
    if (key === UNKNOWN_LABEL.toLowerCase()) return { ok: false, reason: `"${UNKNOWN_LABEL}" is reserved; use the Unknown card.` };
    if (seen.has(key)) return { ok: false, reason: `Two people are named "${c.name.trim()}".` };
    seen.add(key);
  }
  const short = classes.filter((c) => sampleQuality(c).quality === 'too-few');
  if (short.length) {
    const list = short.map((c) => `${c.name} ${c.samples.length}/${sampleQuality(c).min}`).join(', ');
    return { ok: false, reason: `Need at least ${MIN_SAMPLES_PERSON} photos per person and ${MIN_SAMPLES_UNKNOWN} for Unknown (${list}).` };
  }
  if (!ready) return { ok: false, reason: 'Models are still loading.' };
  return { ok: true };
}

/** Current names for the trained outputs (renames after training show up). */
export function resolveLabels(trained: TrainedModel, classes: Person[]): string[] {
  return trained.classIds.map((id, i) => classes.find((c) => c.id === id)?.name.trim() || trained.labels[i]);
}

/** Identity color: persons take palette slots in order; Unknown is neutral gray. */
export function classColor(classes: Person[], id: string): string {
  const cls = classes.find((c) => c.id === id);
  if (!cls || cls.kind === 'unknown') return UNKNOWN_COLOR;
  const slot = classes.filter((c) => c.kind === 'person').findIndex((c) => c.id === id);
  return PERSON_COLORS[slot % PERSON_COLORS.length];
}
