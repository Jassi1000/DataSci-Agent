// frontend/src/api/pipeline.js
import client from './client'

export const runPipeline = (formData) =>
  client.post('/pipeline/run', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)

export const getPipelineStatus = (runId) =>
  client.get(`/pipeline/status/${runId}`).then((r) => r.data)

export const getPipelineResult = (runId) =>
  client.get(`/pipeline/result/${runId}`).then((r) => r.data)

