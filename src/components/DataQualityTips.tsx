import { GOOD_SAMPLES_PERSON, MIN_SAMPLES_PERSON, MIN_SAMPLES_UNKNOWN } from '../config';
import { ChevronIcon } from './ui/icons';

const TIPS: [string, string][] = [
  ['Vary angles', 'front, 45°, profile'],
  ['Vary lighting', 'indoor, outdoor, backlit'],
  ['Vary expressions', 'neutral, smiling, talking'],
  ['Vary distance', 'close and medium'],
  ['Include "Unknown"', 'about 20% of photos should be strangers and other people'],
];

/** Collapsible guidance: diversity matters more than volume. */
export function DataQualityTips({ defaultOpen = false }: { defaultOpen?: boolean }) {
  return (
    <details open={defaultOpen} className="group rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 [&_summary::-webkit-details-marker]:hidden">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-slate-800">
        Data quality tips: 20 varied photos beat 200 identical selfies
        <ChevronIcon size={16} className="text-slate-400 transition-transform group-open:rotate-180" />
      </summary>
      <div className="grid gap-4 border-t border-slate-100 px-4 py-4 text-sm text-slate-600 md:grid-cols-[1fr_1fr]">
        <ul className="flex flex-col gap-1.5">
          {TIPS.map(([title, detail]) => (
            <li key={title} className="flex gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-600" />
              <span>
                <b className="font-semibold text-slate-800">{title}</b>: {detail}
              </span>
            </li>
          ))}
        </ul>
        <div className="flex flex-col gap-2 rounded-xl bg-slate-50 p-3 text-xs leading-relaxed">
          <p>
            <b className="text-slate-800">Requirements:</b> at least {MIN_SAMPLES_PERSON} photos per person ({GOOD_SAMPLES_PERSON}–80 recommended) and{' '}
            {MIN_SAMPLES_UNKNOWN} for Unknown if you add it.
          </p>
          <p>
            Every photo is checked by a face detector: photos with no face or more than one face are skipped, and only the face (plus a 20%
            margin) is kept.
          </p>
          <p>
            Train in the conditions you'll test in. If you'll use the webcam at your desk, include webcam photos from your desk.
          </p>
        </div>
      </div>
    </details>
  );
}
