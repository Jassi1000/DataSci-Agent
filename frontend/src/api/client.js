// frontend/src/api/client.js
import axios from 'axios'

const client = axios.create({ baseURL: '/api' })

export default client

