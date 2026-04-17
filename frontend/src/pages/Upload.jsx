// frontend/src/pages/Upload.jsx
import { useState } from 'react'
import { Check, Loader2, Target, UploadCloud, Zap } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { runPipeline } from '../api/pipeline'
import usePipelineStore from '../store/pipelineStore'
import ErrorCard from '../components/shared/ErrorCard'

const models = [
  ['LogisticRegression', 'Fast linear classifier', 'Classification'],
  ['RandomForestClassifier', 'Robust tree ensemble', 'Classification'],
  ['XGBClassifier', 'Boosted classifier', 'Classification'],
  ['LinearRegression', 'Linear regression baseline', 'Regression'],
  ['RandomForestRegressor', 'Robust regression ensemble', 'Regression'],
  ['XGBRegressor', 'Boosted regressor', 'Regression'],
  ['GradientBoostingClassifier', 'Gradient boosted classifier', 'Classification'],
  ['Ridge', 'Regularized linear regression', 'Regression'],
  ['SVR', 'Support vector regression', 'Regression'],
]

export default function Upload() {
  const navigate = useNavigate()
  const { selectedModels, setSelectedModels, fastMode, setFastMode, setRunId } = usePipelineStore()
  const [file, setFile] = useState(null)
  const [columns, setColumns] = useState([])
  const [preview, setPreview] = useState([])
  const [goal, setGoal] = useState('')
  const [target, setTarget] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const onFile = (picked) => {
    const f = picked?.[0]
    if (!f) return
    setFile(f)
    if (f.name.toLowerCase().endsWith('.csv')) {
      const reader = new FileReader()
      reader.onload = () => {
        const lines = String(reader.result).split(/\r?\n/).filter(Boolean)
        const header = splitCsv(lines[0])
        setColumns(header)
        setTarget(header[header.length - 1] || '')
        setPreview(lines.slice(1, 11).map((line) => splitCsv(line)))
      }
      reader.readAsText(f)
    } else {
      setColumns([])
      setPreview([])
      setTarget('')
    }
  }

  const toggleModel = (name) => setSelectedModels(selectedModels.includes(name) ? selectedModels.filter((m) => m !== name) : [...selectedModels, name])

  const submit = async () => {
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('goal', goal || 'Find the best predictive model and explain the key drivers.')
      form.append('target_col', target)
      form.append('selected_models', JSON.stringify(selectedModels))
      form.append('fast_mode', String(fastMode))
      form.append('run_explainability', 'true')
      const res = await runPipeline(form)
      setRunId(res.run_id)
      navigate(`/dashboard/results/${res.run_id}`)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <StepIndicator step={file ? 2 : 1} />
      {error ? <ErrorCard title="Upload failed" message="The analysis could not be started." detail={error} /> : null}
      <div onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files) }} onDragOver={(e) => e.preventDefault()} className="rounded-2xl border-2 border-dashed border-surface-500 p-12 text-center transition hover:border-brand-500 hover:bg-brand-500/5">
        <UploadCloud className="mx-auto h-12 w-12 text-brand-300" />
        <p className="mt-4 text-lg font-semibold text-white">{file ? file.name : 'Drop CSV or Excel file here'}</p>
        <label className="mt-4 inline-flex cursor-pointer rounded-lg bg-brand-500 px-4 py-2 font-semibold text-white"><input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={(e) => onFile(e.target.files)} />Select File</label>
      </div>
      {file ? (
        <div className="space-y-6">
          {preview.length ? <Preview columns={columns} rows={preview} /> : <div className="rounded-lg bg-surface-700 p-4 text-slate-300">Excel preview is prepared on the backend after upload. Target column can be left blank for automatic detection.</div>}
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-lg bg-surface-700 p-5">
              <label className="text-sm text-slate-300">Goal</label>
              <textarea value={goal} onChange={(e) => setGoal(e.target.value)} className="mt-2 h-28 w-full resize-none rounded-lg bg-surface-800 p-3 text-white outline-none ring-brand-500 focus:ring-2" placeholder="Describe your analysis goal..." />
              <label className="mt-4 block text-sm text-slate-300">Target column</label>
              <select value={target} onChange={(e) => setTarget(e.target.value)} className="mt-2 w-full rounded-lg bg-surface-800 p-3 text-white outline-none">{columns.length ? columns.map((c) => <option key={c}>{c}</option>) : <option value="">Auto-detect</option>}</select>
            </div>
            <div className="rounded-lg bg-surface-700 p-5">
              <p className="text-sm text-slate-300">Mode</p>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <Mode selected={fastMode} onClick={() => setFastMode(true)} icon={Zap} title="Fast Mode" />
                <Mode selected={!fastMode} onClick={() => setFastMode(false)} icon={Target} title="Full Mode" />
              </div>
              <button type="button" disabled={loading || !file} onClick={submit} className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-brand-500 px-4 py-3 font-semibold text-white disabled:opacity-50">{loading ? <Loader2 className="h-5 w-5 animate-spin" /> : null}Run Analysis</button>
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {models.map(([name, desc, tag]) => <button type="button" key={name} onClick={() => toggleModel(name)} className={`rounded-lg border p-4 text-left ${selectedModels.includes(name) ? 'border-brand-500 bg-brand-500/10' : 'border-surface-600 bg-surface-700'}`}><div className="flex items-center gap-2 text-white"><span className={`flex h-5 w-5 items-center justify-center rounded border ${selectedModels.includes(name) ? 'border-brand-500 bg-brand-500' : 'border-surface-500'}`}>{selectedModels.includes(name) ? <Check className="h-3 w-3" /> : null}</span>{name}</div><p className="mt-2 text-sm text-slate-400">{desc}</p><p className="mt-2 text-xs text-brand-300">{tag}</p></button>)}
          </div>
        </div>
      ) : null}
    </div>
  )
}

function splitCsv(line) {
  return line.split(',').map((v) => v.replace(/^"|"$/g, '').trim())
}

function StepIndicator({ step }) {
  return <div className="flex items-center justify-center gap-3">{[1, 2, 3].map((n) => <div key={n} className="flex items-center gap-3"><div className={`flex h-9 w-9 items-center justify-center rounded-full font-semibold text-white ${n < step ? 'bg-emerald-500' : n === step ? 'bg-brand-500' : 'bg-surface-600'}`}>{n < step ? <Check className="h-4 w-4" /> : n}</div>{n < 3 ? <div className={`h-px w-16 ${n < step ? 'bg-emerald-500' : 'bg-surface-600'}`} /> : null}</div>)}</div>
}

function Preview({ columns, rows }) {
  return <div className="overflow-auto rounded-lg border border-slate-200"><table className="min-w-full text-left text-sm"><thead className="bg-white text-slate-900 border-b border-slate-200"><tr>{columns.map((c) => <th key={c} className="p-3">{c}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i} className="odd:bg-white even:bg-slate-50 text-slate-800 hover:bg-indigo-50 hover:text-slate-900 transition-colors border-b border-slate-200">{columns.map((c, j) => <td key={c} className="p-3">{row[j]}</td>)}</tr>)}</tbody></table></div>
}

function Mode({ selected, onClick, icon: Icon, title }) {
  return <button type="button" onClick={onClick} className={`rounded-lg border p-4 text-left ${selected ? 'border-brand-500 bg-brand-500/10' : 'border-surface-600 bg-surface-800'}`}><Icon className="h-5 w-5 text-brand-300" /><p className="mt-2 font-semibold text-white">{title}</p></button>
}

