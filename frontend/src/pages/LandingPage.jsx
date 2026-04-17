// frontend/src/pages/LandingPage.jsx
import { Link } from 'react-router-dom'
import { BarChart2, Brain, Download, MessageSquare, Search, Wand2 } from 'lucide-react'
import { motion } from 'framer-motion'

const features = [
  ['Smart Cleaning', 'Repair missing values, duplicates, encodings, and noisy columns.', Wand2, 'text-brand-500'],
  ['AI EDA', 'Surface distributions, correlations, target balance, and quality signals.', Search, 'text-violet-500'],
  ['Model Comparison', 'Train, tune, rank, and explain the strongest model families.', BarChart2, 'text-emerald-500'],
  ['SHAP Explainability', 'Translate model behavior into feature-level evidence.', Brain, 'text-cyan-500'],
  ['Chat With Data', 'Ask follow-up questions grounded in your analysis run.', MessageSquare, 'text-amber-500'],
  ['Export Reports', 'Generate PDF, Excel, and CSV artifacts for decision reviews.', Download, 'text-rose-500'],
]

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white text-gray-900">
      <nav className="sticky top-0 z-50 border-b border-gray-100 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <Link to="/" className="text-xl font-bold text-brand-600">⚡ InsightForge</Link>
          <div className="hidden gap-8 text-sm text-gray-600 md:flex"><a href="#features" className="hover:text-brand-600">Features</a><a href="#how" className="hover:text-brand-600">How It Works</a><a href="#about" className="hover:text-brand-600">About</a></div>
          <Link to="/dashboard" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white">Open Dashboard</Link>
        </div>
      </nav>
      <section className="bg-white py-32 text-center">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mx-auto max-w-4xl px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-brand-600">AI-Powered Data Science</p>
          <h1 className="mt-4 text-5xl font-bold tracking-tight text-gray-900 md:text-6xl">Turn Raw Data Into <span className="text-brand-600">Decisions. Instantly.</span></h1>
          <p className="mx-auto mt-6 max-w-2xl text-xl text-gray-500">Autonomous cleaning, exploration, model training, explainability, chat, and reporting in one focused workflow.</p>
          <div className="mt-8 flex justify-center gap-3"><Link to="/dashboard/upload" className="rounded-lg bg-brand-600 px-5 py-3 font-semibold text-white">Start Free Analysis</Link><a href="#features" className="rounded-lg border border-gray-300 px-5 py-3 font-semibold text-gray-700">Watch Demo</a></div>
          <div className="mt-8 flex flex-wrap justify-center gap-2">{['⚡ Fast', '🔍 Explainable', '🎯 No-Code', '📊 Interactive', '🏆 Portfolio-Grade'].map((pill) => <span key={pill} className="rounded-full bg-gray-100 px-4 py-1 text-sm text-gray-600">{pill}</span>)}</div>
        </motion.div>
      </section>
      <section id="features" className="bg-gray-50 py-20">
        <div className="mx-auto max-w-7xl px-6">
          <h2 className="text-center text-4xl font-bold">Everything You Need</h2>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {features.map(([title, desc, Icon, color]) => <div key={title} className="rounded-2xl border border-gray-100 bg-white p-6 transition hover:shadow-md"><Icon className={`h-7 w-7 ${color}`} /><h3 className="mt-5 font-semibold">{title}</h3><p className="mt-2 text-sm text-gray-500">{desc}</p></div>)}
          </div>
        </div>
      </section>
      <section className="bg-brand-600 py-20 text-center">
        <h2 className="text-4xl font-bold text-white">Ready to analyze your data?</h2>
        <Link to="/dashboard/upload" className="mt-8 inline-flex rounded-lg bg-white px-5 py-3 font-semibold text-brand-600">Start Free Analysis</Link>
      </section>
    </div>
  )
}

