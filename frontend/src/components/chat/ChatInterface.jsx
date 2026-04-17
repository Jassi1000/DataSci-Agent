// frontend/src/components/chat/ChatInterface.jsx
import { useState } from 'react'
import { Loader2, Send } from 'lucide-react'
import { sendMessage } from '../../api/chat'

const chips = ['What is the best model?', 'Which features matter most?', 'Summarize the findings']

export default function ChatInterface({ runId, runMeta }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const ml = runMeta?.ml_results || {}
  const explain = runMeta?.explainability_results || {}
  const score = ml.problem_type === 'classification' ? ml.best_metrics?.f1_weighted : ml.best_metrics?.r2

  const onSend = async (text = input) => {
    const question = text.trim()
    if (!question || loading) return
    const nextMessages = [...messages, { role: 'user', content: question }]
    setMessages(nextMessages)
    setInput('')
    setLoading(true)
    try {
      const history = messages.map((m) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content }))
      const res = await sendMessage(runId, question, history)
      setMessages([...nextMessages, { role: 'assistant', content: res.answer }])
    } catch (err) {
      setMessages([...nextMessages, { role: 'assistant', content: `I could not answer that yet: ${err.response?.data?.detail || err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] bg-surface-900">
      <aside className="w-[280px] border-r border-surface-700 bg-surface-800 p-5">
        <h2 className="font-semibold text-white">Run {runId}</h2>
        <p className="mt-1 text-sm text-slate-400">{runMeta?.dataset_name || 'Dataset'}</p>
        <div className="mt-5 space-y-3 text-sm">
          <Badge label="Target" value={runMeta?.target_col || runMeta?.plan_results?.detected_target || 'N/A'} />
          <Badge label="Problem" value={ml.problem_type || 'N/A'} />
          <Badge label="Best" value={ml.best_model_name || 'N/A'} />
          <Badge label="Score" value={typeof score === 'number' ? score.toFixed(4) : 'N/A'} />
        </div>
        {explain.top_features?.length ? <div className="mt-6"><p className="mb-2 text-xs uppercase text-slate-500">Top Features</p>{explain.top_features.map((f) => <div key={f} className="mb-2 rounded bg-surface-700 px-3 py-2 text-sm text-slate-200">{f}</div>)}</div> : null}
      </aside>
      <main className="flex flex-1 flex-col">
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {messages.map((m, i) => <div key={i} className={`max-w-2xl rounded-2xl px-4 py-2 ${m.role === 'user' ? 'ml-auto bg-brand-500 text-white' : 'mr-auto bg-surface-700 text-gray-100'}`}>{m.content}</div>)}
          {loading ? <div className="mr-auto flex gap-1 rounded-2xl bg-surface-700 px-4 py-3">{[0, 1, 2].map((i) => <span key={i} className="h-2 w-2 animate-bounce rounded-full bg-brand-400" style={{ animationDelay: `${i * 120}ms` }} />)}</div> : null}
        </div>
        <div className="border-t border-surface-700 bg-surface-800 p-4">
          <div className="mb-3 flex flex-wrap gap-2">{chips.map((chip) => <button key={chip} type="button" onClick={() => onSend(chip)} className="rounded-full bg-surface-700 px-3 py-1 text-xs text-slate-300 hover:text-white">{chip}</button>)}</div>
          <div className="flex gap-3">
            <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} className="flex-1 resize-none rounded-lg bg-surface-700 p-3 text-white outline-none ring-brand-500 focus:ring-2" placeholder="Ask about this analysis..." />
            <button type="button" onClick={() => onSend()} disabled={loading || !input.trim()} className="rounded-lg bg-brand-500 px-4 text-white disabled:opacity-50 transition-all duration-200 hover:bg-brand-600 hover:shadow-lg">{loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}</button>
          </div>
        </div>
      </main>
    </div>
  )
}

function Badge({ label, value }) {
  return <div><p className="text-xs text-slate-500">{label}</p><p className="mt-1 rounded bg-surface-700 px-3 py-2 text-slate-200">{value}</p></div>
}

