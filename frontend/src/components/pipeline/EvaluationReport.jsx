// frontend/src/components/pipeline/EvaluationReport.jsx
import { CheckCircle2 } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import ErrorCard from '../shared/ErrorCard'
import SkeletonLoader from '../shared/SkeletonLoader'

export default function EvaluationReport({ evaluationData, mlData }) {
  if (!evaluationData) return <SkeletonLoader lines={6} height="h-12" />
  if (evaluationData.status === 'error' && !mlData?.best_metrics) return <ErrorCard title="Evaluation failed" message="Model diagnostics could not be generated." detail={evaluationData.error} />
  const bestMetric = mlData?.problem_type === 'classification' ? mlData?.best_metrics?.f1_weighted : mlData?.best_metrics?.r2
  const warnings = evaluationData.status === 'error' ? [evaluationData.error || 'Evaluation metrics were partially unavailable.'] : evaluationData.warnings || []
  const problemType = evaluationData.problem_type || mlData?.problem_type
  return (
    <div className="space-y-6">
      {warnings.length ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          {warnings.map((warning, index) => <p key={index}>{warning}</p>)}
        </div>
      ) : null}
      <div className="rounded-lg bg-surface-700 p-5">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5 text-emerald-500" />
          <h3 className="font-semibold text-white">Model Recommendation</h3>
        </div>
        <p className="mt-2 text-slate-300">{evaluationData.recommendation}</p>
        <p className="mt-2 text-sm text-slate-400">Best model: {mlData?.best_model_name} | Score: {typeof bestMetric === 'number' ? bestMetric.toFixed(4) : 'N/A'}</p>
      </div>
      {problemType === 'classification' ? (
        <>
          <ClassificationMetrics metrics={evaluationData.per_class_metrics || mlData?.best_metrics?.classification_report || {}} />
          <ConfusionMatrix matrix={mlData?.confusion_matrix || mlData?.best_metrics?.confusion_matrix || []} />
          <RocChart curves={evaluationData.roc_curve || {}} />
        </>
      ) : (
        <RegressionChart points={evaluationData.residual_analysis || []} />
      )}
      <div className="rounded-lg bg-surface-700 p-5 text-sm text-slate-300">
        Feature importance is available in the Explainability tab when SHAP completes for the selected model.
      </div>
    </div>
  )
}

function RocChart({ curves }) {
  const entries = Object.entries(curves)
  const data = entries[0] ? entries[0][1].fpr.map((fpr, i) => ({ fpr, tpr: entries[0][1].tpr[i] })) : []
  if (!data.length) return <div className="rounded-lg bg-surface-700 p-5 text-sm text-slate-500">ROC curve is available for binary classification runs when score data can be computed.</div>
  return (
    <div className="rounded-lg bg-surface-700 p-5">
      <h3 className="mb-4 font-semibold text-white">ROC Curve</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid stroke="#e2e8f0" />
          <XAxis dataKey="fpr" tick={{ fill: '#475569', fontSize: 12 }} />
          <YAxis tick={{ fill: '#475569', fontSize: 12 }} />
          <Tooltip />
          <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function RegressionChart({ points }) {
  return (
    <div className="rounded-lg bg-surface-700 p-5">
      <h3 className="mb-4 font-semibold text-white">Predicted vs Actual</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid stroke="#e2e8f0" />
          <XAxis dataKey="actual" name="Actual" tick={{ fill: '#475569', fontSize: 12 }} />
          <YAxis dataKey="predicted" name="Predicted" tick={{ fill: '#475569', fontSize: 12 }} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={points} fill="#6366f1" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

function ClassificationMetrics({ metrics }) {
  const rows = Object.entries(metrics).filter(([, value]) => value && typeof value === 'object' && 'precision' in value)
  if (!rows.length) return null
  return (
    <div className="overflow-hidden rounded-lg bg-surface-700">
      <h3 className="p-5 font-semibold text-white">Precision / Recall / F1</h3>
      <table className="w-full text-left text-sm">
        <thead><tr><th className="p-3">Class</th><th className="p-3">Precision</th><th className="p-3">Recall</th><th className="p-3">F1</th><th className="p-3">Support</th></tr></thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-t border-surface-600">
              <td className="p-3 font-medium text-white">{label}</td>
              <td className="p-3 text-slate-300">{fmt(value.precision)}</td>
              <td className="p-3 text-slate-300">{fmt(value.recall)}</td>
              <td className="p-3 text-slate-300">{fmt(value['f1-score'])}</td>
              <td className="p-3 text-slate-300">{value.support}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ConfusionMatrix({ matrix }) {
  if (!matrix?.length) return null
  return (
    <div className="rounded-lg bg-surface-700 p-5">
      <h3 className="mb-4 font-semibold text-white">Confusion Matrix</h3>
      <div className="inline-block overflow-hidden rounded-lg border border-surface-600">
        {matrix.map((row, i) => <div key={i} className="flex">{row.map((cell, j) => <div key={j} className={`flex h-16 w-16 items-center justify-center border border-surface-600 font-semibold ${i === j ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'}`}>{cell}</div>)}</div>)}
      </div>
    </div>
  )
}

function fmt(value) {
  return typeof value === 'number' ? value.toFixed(4) : value ?? 'N/A'
}
