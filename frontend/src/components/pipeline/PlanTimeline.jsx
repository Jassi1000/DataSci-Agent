// frontend/src/components/pipeline/PlanTimeline.jsx
import ErrorCard from '../shared/ErrorCard'
import SkeletonLoader from '../shared/SkeletonLoader'

const colors = {
  data_cleaning: 'border-indigo-400',
  eda: 'border-violet-400',
  ml_training: 'border-emerald-400',
  explainability: 'border-cyan-400',
  evaluation: 'border-amber-400',
  reporting: 'border-rose-400',
}

export default function PlanTimeline({ planData }) {
  if (!planData) return <SkeletonLoader lines={5} height="h-14" />
  if (planData.status === 'error') return <ErrorCard title="Plan failed" message="The planner could not build the workflow." detail={planData.error} />
  return (
    <div className="space-y-0">
      {(planData.plan || []).map((step, index) => (
        <div key={step.step} className="relative flex gap-4 pb-6">
          {index < (planData.plan || []).length - 1 ? <div className="absolute left-5 top-10 h-full w-px bg-surface-600" /> : null}
          <div className="z-10 flex h-10 w-10 items-center justify-center rounded-full bg-brand-500 font-semibold text-white">{step.step}</div>
          <div className={`flex-1 rounded-lg border-l-4 bg-surface-700 p-5 ${colors[step.agent] || 'border-brand-500'}`}>
            <p className="text-sm font-semibold uppercase tracking-wide text-brand-300">{step.agent.replace('_', ' ')}</p>
            <p className="mt-2 text-white">{step.action}</p>
            <p className="mt-2 text-sm text-slate-400">{step.reasoning}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

