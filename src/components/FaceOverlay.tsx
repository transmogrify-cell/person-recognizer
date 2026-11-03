import type { FaceBox } from '../types';

/**
 * Face boxes positioned over an image/video that is displayed at the same
 * aspect ratio as its source (`width` x `height`). `mirror` matches a
 * CSS-mirrored selfie video.
 */
export function FaceOverlay({
  boxes,
  width,
  height,
  mirror = false,
  tone = 'ok',
  label,
}: {
  boxes: FaceBox[];
  width: number;
  height: number;
  mirror?: boolean;
  tone?: 'ok' | 'warn' | 'unknown';
  /** Label for the first (classified) box. */
  label?: string;
}) {
  const colors = { ok: 'border-emerald-400', warn: 'border-amber-400', unknown: 'border-slate-300' };
  const leftPct = (b: FaceBox) => ((mirror ? width - b.xMin - b.width : b.xMin) / width) * 100;
  const main = boxes[0];
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden="true">
      {boxes.map((b, i) => (
        <div
          key={i}
          className={`absolute rounded-md border-2 ${i === 0 ? colors[tone] : 'border-dashed border-white/60'}`}
          style={{ left: `${leftPct(b)}%`, top: `${(b.yMin / height) * 100}%`, width: `${(b.width / width) * 100}%`, height: `${(b.height / height) * 100}%` }}
        />
      ))}
      {main && label && (
        // Kept inside the frame even when the face touches an edge.
        <span
          className="absolute whitespace-nowrap rounded bg-slate-900/80 px-1.5 py-0.5 text-xs font-semibold text-white"
          style={{
            left: `${Math.min(Math.max(leftPct(main), 1), 70)}%`,
            top: `${Math.max((main.yMin / height) * 100 - 9, 1)}%`,
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}
