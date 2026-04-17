// frontend/src/api/chat.js
import client from './client'

export const sendMessage = (runId, question, history) =>
  client.post(`/chat/${runId}`, { question, history }).then((r) => r.data)

