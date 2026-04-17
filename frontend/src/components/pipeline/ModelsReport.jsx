// frontend/src/components/pipeline/ModelsReport.jsx
import { useState } from 'react'
import { Crown } from 'lucide-react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ErrorCard from '../shared/ErrorCard'
import SkeletonLoader from '../shared/SkeletonLoader'

export default function ModelsReport({ mlData }) {
  const [expanded, setExpanded] = useState(null)
  if (!mlData) return <SkeletonLoader lines={8} height="h-12" />
  if (mlData.status === 'error') return <ErrorCard title="Model training failed" message="No model could be trained successfully." detail={mlData.error} />
  const primary = mlData.problem_type === 'classification' ? 'f1_weighted' : 'r2'
  const models = mlData.all_models || []
  const chartData = models.map((m) => ({ name: m.display_name || m.name, value: Number(((m.tuned_metrics || m.metrics || {})[primary]) || 0), error: !!m.error, rank: m.rank }))
  return (
    <div className="space-y-6">
      <div className="overflow-hidden rounded-lg bg-surface-700">
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 border-b border-slate-200"><tr><th className="p-3 text-slate-900 font-semibold">Rank</th><th className="p-3 text-slate-900 font-semibold">Model</th><th className="p-3 text-slate-900 font-semibold">Primary Metric</th><th className="p-3 text-slate-900 font-semibold">F1</th><th className="p-3 text-slate-900 font-semibold">Train Time</th><th className="p-3 text-slate-900 font-semibold">Tuned</th><th className="p-3 text-slate-900 font-semibold">Status</th></tr></thead>
          <tbody>
            {models.map((model) => {
              const metrics = model.tuned_metrics && Object.keys(model.tuned_metrics).length ? model.tuned_metrics : model.metrics || {}
              const rankClass = model.rank === 1 ? 'bg-emerald-500/10' : model.rank === 2 ? 'bg-brand-500/10' : model.rank === 3 ? 'bg-amber-500/10' : ''
              return (
                <tr key={model.name} onClick={() => setExpanded(expanded === model.name ? null : model.name)} className={`cursor-pointer border-t border-surface-600 ${rankClass}`}>
                  <td className="p-3 text-white">{model.rank || '-'}</td>
                  <td className="p-3 text-white">{model.name}{expanded === model.name ? <MetricsGrid metrics={metrics} /> : null}</td>
                  <td className="p-3 text-slate-700">{format(metrics[primary])}</td>
                  <td className="p-3 text-slate-700">{format(metrics.f1_weighted)}</td>
                  <td className="p-3 text-slate-700">{format(model.training_time_sec)}s</td>
                  <td className="p-3">{model.tuned ? <span className="rounded bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-700">TUNED</span> : <span className="text-slate-500">No</span>}</td>
                  <td className="p-3">{model.error ? <span title={model.error} className="rounded bg-red-100 px-2 py-1 text-xs font-semibold text-red-700">FAILED</span> : <span className="rounded bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700">OK</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {models.filter((m) => m.rank && m.rank <= 3).sort((a, b) => a.rank - b.rank).map((model) => <TuningCard key={model.name} model={model} primary={primary} />)}
      </div>
      <div className="rounded-lg border-l-4 border-brand-500 bg-surface-700 p-5">
        <h3 className="text-lg font-semibold text-white">Why It Won</h3>
        <p className="mt-2 text-slate-300">{mlData.why_best}</p>
        <div className="mt-6 h-[300px]">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 12 }} />
              <YAxis tick={{ fill: '#475569', fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value">
                {chartData.map((entry, index) => <Cell key={index} fill={entry.error ? '#475569' : entry.rank === 1 ? '#10b981' : '#6366f1'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      {mlData.problem_type === 'classification' && mlData.confusion_matrix?.length ? <ConfusionMatrix matrix={mlData.confusion_matrix} /> : null}
      <div className="rounded-lg bg-slate-50 border border-slate-200 p-5 text-sm text-slate-700">
        <h3 className="mb-3 font-semibold text-slate-900">Preprocessing Notes</h3>
        <p>{mlData.preprocessing_notes}</p>
        <p className="mt-2">Scaler: {mlData.scaler_used} | SMOTE: {mlData.smote_applied ? 'Applied' : 'Not applied'} | Train/Test: {mlData.train_size}/{mlData.test_size} | Imbalance: {mlData.imbalance_ratio}</p>
      </div>
    </div>
  )
}

function TuningCard({ model, primary }) {
  const before = model.metrics?.[primary]
  const after = model.tuned_metrics?.[primary] || before
  const delta = Number(after || 0) - Number(before || 0)
  return (
    <div className="rounded-lg bg-surface-700 p-5">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-white">{model.name}</h3>
        {model.rank === 1 ? <span className="flex items-center gap-1 rounded bg-emerald-500/20 px-2 py-1 text-xs text-emerald-200"><Crown className="h-3 w-3" /> CHAMPION</span> : null}
      </div>
      <p className="mt-3 text-sm text-slate-300">{format(before)} → {format(after)} <span className="text-emerald-300">+{delta.toFixed(3)}</span></p>
      <pre className="mt-3 max-h-40 overflow-auto rounded-lg bg-surface-900 p-3 text-xs text-slate-300">{JSON.stringify(model.tuned_params || {}, null, 2)}</pre>
    </div>
  )
}

function MetricsGrid({ metrics }) {
  return <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-300">{Object.entries(metrics).filter(([, v]) => typeof v !== 'object').map(([k, v]) => <span key={k} className="rounded bg-surface-900 px-2 py-1">{k}: {format(v)}</span>)}</div>
}

function ConfusionMatrix({ matrix }) {
  return (
    <div className="rounded-lg bg-surface-700 p-5">
      <h3 className="mb-4 font-semibold text-white">Confusion Matrix</h3>
      <div className="inline-block overflow-hidden rounded-lg border border-surface-600">
        {matrix.map((row, i) => <div key={i} className="flex">{row.map((cell, j) => <div key={j} className={`flex h-16 w-16 items-center justify-center border border-surface-600 font-semibold text-white ${i === j ? 'bg-emerald-500/30' : 'bg-red-500/30'}`}>{cell}</div>)}</div>)}
      </div>
    </div>
  )
}

function format(value) {
  return typeof value === 'number' ? value.toFixed(4) : value ?? '-'
}
