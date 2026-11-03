import { useEffect } from 'react';
import { AccuracyReport } from './components/AccuracyReport';
import { AppHeader } from './components/AppHeader';
import { InferencePanel } from './components/InferencePanel';
import { Landing } from './components/Landing';
import { TrainingProgress } from './components/TrainingProgress';
import { Workspace } from './components/Workspace';
import { Toaster } from './components/ui/Toaster';
import { useFaceDetector } from './hooks/useFaceDetector';
import { useProjectStore } from './store/useProjectStore';

export default function App() {
  useFaceDetector(); // load BlazeFace + MobileNet right away
  const screen = useProjectStore((s) => s.screen);

  // Photos live only in memory: warn before a reload throws them away.
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      const { classes, trained } = useProjectStore.getState();
      if (classes.some((c) => c.samples.length > 0) || trained) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  return (
    <>
      {screen === 'landing' ? (
        <Landing />
      ) : (
        <div className="min-h-screen">
          <AppHeader />
          {screen === 'workspace' && <Workspace />}
          {screen === 'training' && <TrainingProgress />}
          {screen === 'report' && <AccuracyReport />}
          {screen === 'inference' && <InferencePanel />}
        </div>
      )}
      <Toaster />
    </>
  );
}
