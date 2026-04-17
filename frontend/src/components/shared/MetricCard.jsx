// frontend/src/components/shared/MetricCard.jsx
export default function MetricCard({ label, value, icon: Icon, color = 'brand', delta }) {
  const colorMap = {
    brand: 'bg-brand-500/15 text-brand-100',
    red: 'bg-red-500/15 text-red-200',
    amber: 'bg-amber-500/15 text-amber-200',
    blue: 'bg-sky-500/15 text-sky-200',
    purple: 'bg-violet-500/15 text-violet-200',
    emerald: 'bg-emerald-500/15 text-emerald-200',
  }
  const deltaPositive = String(delta || '').startsWith('+')
  return (
    <div className="rounded-lg bg-surface-700 p-5 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-400">{label}</p>
          <p className="mt-2 text-3xl font-semibold text-white">{value}</p>
        </div>
        {Icon ? (
          <div className={`rounded-lg p-3 ${colorMap[color] || colorMap.brand}`}>
            <Icon className="h-5 w-5" />
          </div>
        ) : null}
      </div>
      {delta ? <span className={`mt-4 inline-flex rounded-full px-2 py-1 text-xs ${deltaPositive ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'}`}>{delta}</span> : null}
    </div>
  )
}

