import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

export const getPatients = async (limit = 50) => {
  const res = await api.get(`/patients?limit=${limit}`)
  return res.data
}

export const getPatient = async (id: string) => {
  const res = await api.get(`/patients/${id}`)
  return res.data
}

export const getSeries = async (id: string, hours = 72) => {
  const res = await api.get(`/patients/${id}/series?hours=${hours}`)
  return res.data
}

export const getHealth = async () => {
  const res = await api.get('/health')
  return res.data
}

export default api
