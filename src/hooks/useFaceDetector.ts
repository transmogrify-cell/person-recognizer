import { useCallback, useEffect } from 'react';
import { loadFaceDetector } from '../ml/faceDetector';
import { initBackend, loadMobilenet } from '../ml/mobilenet';
import { useProjectStore } from '../store/useProjectStore';
import { toast } from '../store/useToastStore';

/**
 * Loads both pre-trained models on app start: BlazeFace (face detection, needed
 * as soon as photos are added) and MobileNet (features, needed for training).
 */
export function useFaceDetector() {
  const detector = useProjectStore((s) => s.detector);
  const mobilenet = useProjectStore((s) => s.mobilenet);
  const backend = useProjectStore((s) => s.backend);

  const load = useCallback(async () => {
    const { setModelStatus, setBackend } = useProjectStore.getState();
    const b = await initBackend();
    setBackend(b);
    if (b !== 'webgl') toast.warning('WebGL is not available, so the slow CPU backend is used.');

    const track = async (which: 'detector' | 'mobilenet', run: () => Promise<unknown>) => {
      if (useProjectStore.getState()[which].status === 'ready') return;
      setModelStatus(which, { status: 'loading' });
      try {
        await run();
        setModelStatus(which, { status: 'ready' });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setModelStatus(which, { status: 'error', error: message });
        toast.error(message);
      }
    };
    // Face detector first: it's small and gates adding photos.
    await track('detector', loadFaceDetector);
    await track('mobilenet', loadMobilenet);
  }, []);

  useEffect(() => {
    const s = useProjectStore.getState();
    if (s.detector.status === 'idle' && s.mobilenet.status === 'idle') void load();
  }, [load]);

  return { detector, mobilenet, backend, retry: load };
}
