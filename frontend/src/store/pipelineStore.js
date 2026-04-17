// frontend/src/store/pipelineStore.js
import { create } from 'zustand'

const usePipelineStore = create((set) => ({
  runId: null,
  status: null,
  result: null,
  selectedModels: ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier'],
  fastMode: false,
  setRunId: (id) => set({ runId: id }),
  setStatus: (s) => set({ status: s }),
  setResult: (r) => set({ result: r }),
  setSelectedModels: (m) => set({ selectedModels: m }),
  setFastMode: (f) => set({ fastMode: f }),
  reset: () => set({ runId: null, status: null, result: null }),
}))

export default usePipelineStore

