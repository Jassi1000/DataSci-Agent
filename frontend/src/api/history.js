// frontend/src/api/history.js
import client from './client'

export const getHistory = () => client.get('/history').then((r) => r.data)

export const deleteRun = (runId) =>
  client.delete(`/history/${runId}`).then((r) => r.data)

