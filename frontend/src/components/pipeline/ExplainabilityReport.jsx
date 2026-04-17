// frontend/src/components/pipeline/ExplainabilityReport.jsx
import { Lightbulb, TrendingDown, TrendingUp } from 'lucide-react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ErrorCard from '../shared/ErrorCard'
import SkeletonLoader from '../shared/SkeletonLoader'

export default function ExplainabilityReport({ explainData }) {
  if (!explainData) return <SkeletonLoader lines={6} height="h-12" />
  if (explainData.status === 'skipped') return <div className="rounded-lg bg-surface-700 p-5 text-slate-300">Explainability skipped or unavailable: {explainData.reason}</div>
  if (explainData.status === 'error') return <ErrorCard title="Explainability failed" message="SHAP could not explain this model." detail={explainData.error} />
  const data = (explainData.global_importance || []).slice(0, 15)
  return (
    <div className="space-y-6">
      <div className="flex gap-3 rounded-lg bg-surface-700 p-5">
        <Lightbulb className="h-6 w-6 text-amber-300" />
        <p className="text-white">{explainData.plain_english}</p>
      </div>
      <div className="rounded-lg bg-surface-700 p-5">
        <h3 className="mb-4 font-semibold text-white">Global Importance</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={data} layout="vertical">
            <XAxis type="number" tick={{ fill: '#475569', fontSize: 12 }} />
            <YAxis type="category" dataKey="feature" width={130} tick={{ fill: '#475569', fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="mean_abs_shap" fill="#6366f1" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {data.slice(0, 3).map((item, index) => (
          <div key={item.feature} className="rounded-lg bg-surface-700 p-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white">{item.feature}</h3>
              {index === 0 ? <TrendingUp className="h-5 w-5 text-emerald-300" /> : <TrendingDown className="h-5 w-5 text-red-300" />}
            </div>
            <p className="mt-3 text-2xl font-bold text-brand-300">{Number(item.mean_abs_shap).toFixed(4)}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
