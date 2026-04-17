// frontend/src/components/pipeline/EDAReport.jsx
import { useState } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import ErrorCard from '../shared/ErrorCard'
import SkeletonLoader from '../shared/SkeletonLoader'

function ChartRenderer({ chart, title }) {
  const [failed, setFailed] = useState(false)
  if (!chart || failed) return <div className="p-4 text-sm text-slate-500">Chart unavailable</div>
  if (chart?.type === 'unavailable') return <div className="p-4 text-sm text-slate-500">{chart.reason || 'Chart generation requires kaleido'}</div>
  const src = typeof chart === 'string' ? `data:image/png;base64,${chart}` : chart?.data ? `data:image/png;base64,${chart.data}` : null
  if (src) return <img src={src} alt={title} className="w-full rounded-lg" onError={() => setFailed(true)} />
  return <div className="p-4 text-sm text-slate-500">Chart format not supported</div>
}

export default function EDAReport({ edaData }) {
  if (!edaData) return <SkeletonLoader lines={8} height="h-12" />
  if (edaData.status === 'error') return <ErrorCard title="EDA failed" message="Exploratory analysis could not complete." detail={edaData.error} />
  const dtypes = edaData.dtypes || {}
  const missingTotal = Object.values(edaData.missing_per_column || {}).reduce((a, b) => a + Number(b || 0), 0)
  const cells = Math.max((edaData.shape?.[0] || 0) * (edaData.shape?.[1] || 0), 1)
  const classData = Object.entries(edaData.class_balance || {}).map(([name, v]) => ({ name, value: v.count }))
  const firstDist = Object.values(edaData.charts?.distributions || {})[0]
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-emerald-900">
        <CheckCircle2 className="h-5 w-5 text-emerald-600" />
        Recommended target: <strong>{edaData.recommended_target || 'Not detected'}</strong>
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Pill label="Shape" value={(edaData.shape || []).join(' x ')} />
        <Pill label="Missing" value={`${((missingTotal / cells) * 100).toFixed(2)}%`} />
        <Pill label="Numeric Cols" value={Object.values(dtypes).filter((v) => /int|float/.test(v)).length} />
        <Pill label="Categorical Cols" value={Object.values(dtypes).filter((v) => !/int|float/.test(v)).length} />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {(edaData.insights || []).map((insight, i) => <Insight key={i} insight={insight} />)}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Correlation Matrix"><ChartRenderer chart={edaData.charts?.correlation_matrix} title="Correlation Matrix" /></ChartCard>
        <ChartCard title="Box Plots"><ChartRenderer chart={edaData.charts?.boxplots} title="Box Plots" /></ChartCard>
        <ChartCard title="Missing Values"><ChartRenderer chart={edaData.charts?.missing_heatmap} title="Missing Values" /></ChartCard>
        <ChartCard title="Distribution"><ChartRenderer chart={firstDist} title="Distribution" /></ChartCard>
      </div>
      {classData.length ? (
        <ChartCard title="Class Balance">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={classData} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110}>
                {classData.map((_, i) => <Cell key={i} fill={['#6366f1', '#10b981', '#f59e0b', '#f43f5e'][i % 4]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      ) : null}
    </div>
  )
}

function Pill({ label, value }) {
  return <div className="rounded-lg bg-surface-700 p-4"><p className="text-xs text-slate-400">{label}</p><p className="mt-1 text-lg font-semibold text-white">{value}</p></div>
}

function ChartCard({ title, children }) {
  return <div className="rounded-lg bg-surface-700 p-5"><h3 className="mb-4 font-semibold text-white">{title}</h3>{children}</div>
}

function Insight({ insight }) {
  const styles = { HIGH: 'border-red-400 text-red-200 bg-red-500/10', MEDIUM: 'border-amber-400 text-amber-200 bg-amber-500/10', LOW: 'border-emerald-400 text-emerald-200 bg-emerald-500/10' }
  return (
    <div className={`rounded-lg border-t-4 bg-surface-700 p-5 ${styles[insight.severity] || styles.LOW}`}>
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-semibold text-white">{insight.topic}</h3>
        <span className="rounded-full px-2 py-1 text-xs font-semibold">{insight.severity}</span>
      </div>
      <p className="mt-2 text-sm text-slate-400">{insight.description}</p>
    </div>
  )
}

