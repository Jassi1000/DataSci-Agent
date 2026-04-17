// frontend/src/components/pipeline/CleaningReport.jsx
import { useState } from 'react'
import { Database, Eraser, Rows, ShieldCheck } from 'lucide-react'
import ErrorCard from '../shared/ErrorCard'
import MetricCard from '../shared/MetricCard'
import SkeletonLoader from '../shared/SkeletonLoader'

function QualityScore({ score = 0 }) {
  const color = score > 80 ? 'text-emerald-400' : score >= 60 ? 'text-amber-400' : 'text-red-400'
  const dash = 283 - (283 * score) / 100
  return (
    <div className="relative h-36 w-36">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="none" stroke="#252836" strokeWidth="8" />
        <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" strokeDasharray="283" strokeDashoffset={dash} className={color} strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-bold ${color}`}>{Math.round(score)}</span>
        <span className="text-xs text-slate-400">Quality</span>
      </div>
    </div>
  )
}

export default function CleaningReport({ cleaningData }) {
  const [open, setOpen] = useState(true)
  if (!cleaningData) return <SkeletonLoader lines={6} height="h-12" />
  if (cleaningData.status === 'error') return <ErrorCard title="Cleaning failed" message="The data cleaning agent could not complete." detail={cleaningData.error} />
  const types = cleaningData.column_types || {}
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <MetricCard label="Rows Removed" value={cleaningData.rows_removed || 0} icon={Rows} color="red" />
        <MetricCard label="Cols Removed" value={cleaningData.cols_removed || 0} icon={Database} color="amber" />
        <MetricCard label="Missing Fixed" value={cleaningData.missing_fixed || 0} icon={ShieldCheck} color="blue" />
        <MetricCard label="Duplicates Removed" value={cleaningData.duplicates_removed || 0} icon={Eraser} color="purple" />
      </div>
      <div className="grid gap-6 lg:grid-cols-[180px_1fr]">
        <div className="rounded-lg bg-surface-700 p-5">
          <QualityScore score={cleaningData.data_quality_score || 0} />
        </div>
        <div className="rounded-lg bg-surface-700 p-5">
          <button type="button" onClick={() => setOpen((v) => !v)} className="mb-4 font-semibold text-white">Cleaning Log</button>
          {open ? (
            <ul className="space-y-2">
              {(cleaningData.cleaning_log || []).map((item, index) => (
                <li key={index} className="flex gap-3 text-sm text-slate-300">
                  <span className="mt-2 h-2 w-2 rounded-full bg-brand-400" />
                  {item}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
      <div className="overflow-hidden rounded-lg bg-white border border-slate-200">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr><th className="p-3 text-slate-900 font-semibold">Column</th><th className="p-3 text-slate-900 font-semibold">Original Type</th><th className="p-3 text-slate-900 font-semibold">New Type</th></tr>
          </thead>
          <tbody>
            {Object.entries(types).map(([col, value]) => (
              <tr key={col} className="border-t border-slate-200 odd:bg-white even:bg-slate-50 hover:bg-slate-100">
                <td className="p-3 text-slate-900 font-medium">{col}</td><td className="p-3 text-slate-700">{value.original}</td><td className="p-3 text-slate-700">{value.cleaned}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
