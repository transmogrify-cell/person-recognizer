import { useMemo, useState } from 'react';
import { buildMetadata, exportTfjs, kerasPythonSnippet } from '../ml/exporter';
import { resolveLabels, useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';
import { Button } from './ui/Button';
import { CodeBlock } from './ui/CodeBlock';
import { DownloadIcon, InfoIcon, SpinnerIcon } from './ui/icons';
import { Modal } from './ui/Modal';
import { Tabs, type TabItem } from './ui/Tabs';

type ExportTab = 'tfjs' | 'keras';
const TABS: TabItem<ExportTab>[] = [
  { id: 'tfjs', label: 'TensorFlow.js' },
  { id: 'keras', label: 'Keras (.h5)' },
];

const LOAD_SNIPPET = `import { loadModel } from './ml/predictor';

// The downloaded .zip from an <input type="file">, or the unzipped folder:
const model = await loadModel(zipFile);
// const model = await loadModel({ modelUrl: '/my-model/model.json' });

// Face detection, cropping, features and gating all happen inside predict():
const result = await model.predict(imageFile);
// { person: 'Alice', confidence: 0.94, allProbs: { Alice: 0.94, Bob: 0.03, Unknown: 0.03 }, faceCount: 1 }
// person is 'Unknown / Not sure' below the threshold, and null when no face is found.`;

export function ExportModal({ open, onClose }: { open: boolean; onClose(): void }) {
  const trained = useProjectStore((s) => s.trained);
  const classes = useProjectStore((s) => s.classes);
  const threshold = useProjectStore((s) => s.threshold);
  const [tab, setTab] = useState<ExportTab>('tfjs');
  const [working, setWorking] = useState(false);
  const labels = useMemo(() => (trained ? resolveLabels(trained, classes) : []), [trained, classes]);
  const metadata = useMemo(() => (trained ? buildMetadata(trained, labels, threshold) : null), [trained, labels, threshold]);

  const download = async () => {
    if (!trained) return;
    setWorking(true);
    try {
      await exportTfjs(trained, labels, threshold);
      toast.success('Model exported.');
    } catch (err) {
      toast.error(`Export failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setWorking(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Export your model">
      <Tabs<ExportTab> tabs={TABS} value={tab} onChange={setTab}>
        <div className="flex flex-col gap-5 p-5 text-sm text-slate-600">
          {tab === 'tfjs' ? (
            <>
              <p>
                A <code className="rounded bg-slate-100 px-1">.zip</code> with the trained head and its metadata. BlazeFace and MobileNet are not
                included (they're identical for every model) and are re-attached when you load it.
              </p>
              <ul className="divide-y divide-slate-100 rounded-xl ring-1 ring-slate-200">
                {[
                  ['model.json', 'head architecture + weight manifest'],
                  ['weights.bin', 'trained head weights (~1.4 MB)'],
                  ['metadata.json', 'labels, face-crop settings, threshold'],
                  ['convert_to_keras.py', 'the Keras script from the next tab'],
                ].map(([name, desc]) => (
                  <li key={name} className="flex flex-col gap-0.5 px-3 py-2 sm:flex-row sm:gap-3">
                    <code className="shrink-0 font-mono text-xs font-semibold text-slate-800 sm:w-40">{name}</code>
                    <span className="text-xs text-slate-500">{desc}</span>
                  </li>
                ))}
              </ul>
              <div>
                <Button variant="primary" onClick={() => void download()} disabled={!trained || working}>
                  {working ? <SpinnerIcon size={16} /> : <DownloadIcon size={16} />}
                  Download TensorFlow.js model
                </Button>
              </div>
              {metadata && <CodeBlock label="metadata.json" code={JSON.stringify(metadata, null, 2)} maxHeight={220} />}
              <CodeBlock label="load it again with loadModel()" code={LOAD_SNIPPET} />
            </>
          ) : (
            <>
              <p className="flex gap-2 rounded-xl bg-brand-50 px-3 py-2.5 text-[13px] leading-relaxed text-brand-900">
                <InfoIcon size={16} className="mt-0.5 shrink-0" />
                <span>
                  Browsers can't write HDF5, so this script rebuilds the <b>head</b> in tf.keras from the TF.js export and saves{' '}
                  <code>person_head.h5</code>. The MobileNet base (and the face crop) must be applied separately: the head's input is the 1280-d
                  feature vector of a face crop, not an image.
                </span>
              </p>
              <CodeBlock
                label="run locally"
                code={`unzip person-model-tfjs.zip -d person-model && cd person-model
pip install "tensorflow>=2.12"
python convert_to_keras.py      # -> person_head.h5 + labels.txt`}
              />
              <CodeBlock label="convert_to_keras.py" code={kerasPythonSnippet()} maxHeight={380} />
            </>
          )}
        </div>
      </Tabs>
    </Modal>
  );
}
