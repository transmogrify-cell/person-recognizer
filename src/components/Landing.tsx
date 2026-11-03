import { useRef, useState, type ChangeEvent } from 'react';
import { loadModel } from '../ml/predictor';
import { modelsReady, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { Logo, ModelStatus } from './AppHeader';
import { FolderIcon, ScanIcon, ShieldIcon, SpinnerIcon } from './ui/icons';

/** Screen 1. */
export function Landing() {
  const goTo = useProjectStore((s) => s.goTo);
  const loadTrained = useProjectStore((s) => s.loadTrained);
  const ready = useProjectStore(modelsReady);
  const hasWork = useProjectStore((s) => s.classes.some((c) => c.samples.length > 0) || s.trained !== null);
  const [opening, setOpening] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const openModel = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = '';
    if (!files.length) return;
    setOpening(true);
    try {
      const loaded = await loadModel(files.length === 1 && /\.zip$/i.test(files[0].name) ? files[0] : files);
      loadTrained(
        {
          head: loaded.head,
          labels: loaded.labels,
          kinds: loaded.kinds,
          featureDim: loaded.metadata.featureDim,
          trainedAt: Date.parse(loaded.metadata.createdAt) || Date.now(),
          sampleCount: 0,
          evaluation: null,
        },
        loaded.metadata.confidenceThreshold,
      );
      toast.success(`Opened a model for ${loaded.labels.join(', ')}.`);
    } catch (err) {
      toast.error(`Could not open the model: ${err instanceof Error ? err.message : err}`);
    } finally {
      setOpening(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-10 sm:py-16">
      <header className="mb-10 flex items-center gap-2.5">
        <Logo />
        <span className="text-lg font-semibold text-slate-900">Person Recognizer</span>
      </header>

      <button
        type="button"
        onClick={() => goTo('workspace')}
        className="group flex flex-col items-start gap-5 rounded-3xl bg-white p-8 text-left shadow-sm ring-1 ring-slate-200 transition hover:-translate-y-0.5 hover:shadow-md hover:ring-brand-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500 sm:p-10"
      >
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600 text-white">
          <ScanIcon size={28} />
        </span>
        <span>
          <span className="block text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">Person Recognition Trainer</span>
          <span className="mt-3 block max-w-2xl text-lg text-slate-600">
            Teach your browser to recognise the people you choose. Add photos or webcam captures of each person, train in seconds, and get an
            honest accuracy report. Nothing leaves this tab.
          </span>
        </span>
        <span className="flex flex-wrap gap-2 text-xs font-medium text-slate-600">
          {['Face detection + crop', '6× augmentation', 'Per-person precision & recall', 'Confusion matrix', '"Unknown" rejection'].map((t) => (
            <span key={t} className="rounded-full bg-slate-100 px-2.5 py-1">
              {t}
            </span>
          ))}
        </span>
        <span className="text-base font-semibold text-brand-700 group-hover:underline">{hasWork ? 'Continue project →' : 'Get started →'}</span>
      </button>

      <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          disabled={opening || !ready}
          className="inline-flex items-center gap-2 font-medium text-slate-700 hover:text-brand-700 disabled:text-slate-400"
        >
          {opening ? <SpinnerIcon size={16} /> : <FolderIcon size={16} />}
          Open an exported model
        </button>
        <input ref={fileInput} type="file" accept=".zip,.json,.bin" multiple hidden onChange={openModel} />
        <span className="text-xs">
          <ModelStatus />
        </span>
      </div>

      <section className="mt-10 rounded-2xl bg-white p-5 text-sm leading-relaxed text-slate-600 ring-1 ring-slate-200">
        <h2 className="mb-2 flex items-center gap-2 font-semibold text-slate-800">
          <ShieldIcon size={16} /> What this is (and isn't)
        </h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            A frozen-feature + softmax classifier, <b>not</b> FaceNet/ArcFace. It recognises the people you train it on and won't generalise to
            "any person".
          </li>
          <li>Accuracy depends on variety: 20 good, varied photos beat 200 near-identical selfies.</li>
          <li>Lighting and pose that differ between training and testing hurt accuracy the most.</li>
          <li>WebGL (GPU) is recommended; the CPU fallback is slow.</li>
          <li>A learning/demo tool. Not suitable for security or authentication.</li>
        </ul>
      </section>
    </div>
  );
}
