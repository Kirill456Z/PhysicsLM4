import type { SampleResponse, BatchResponse, ValidateResponse } from '../types'

const API_BASE = '/api'

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(API_BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    const msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

export const api = {
  getTasks: () => fetchJson<{ tasks: string[] }>('/tasks'),

  getDefaultConfig: (taskName: string) =>
    fetchJson<{ task_name: string; config_yaml: string }>(`/default-config/${taskName}`),

  validateConfig: (configYaml: string) =>
    fetchJson<ValidateResponse>('/validate-config', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: configYaml }),
    }),

  generateSample: (configYaml: string, taskName: string) =>
    fetchJson<SampleResponse>('/generate-sample', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: configYaml, task_name: taskName }),
    }),

  generateBatch: (configYaml: string, taskName: string, batchSize: number) =>
    fetchJson<BatchResponse>('/generate-batch', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: configYaml, task_name: taskName, batch_size: batchSize }),
    }),

  evaluate: (payload: {
    config_yaml: string
    task_name: string
    generation: number[]
    task_index: number
    context: number[]
    loss_mask: number[]
    answer_start_index: number
    query_nodes: number[][]
    answer_nodes: number[][]
    num_hops: number[]
    graph_nodes: number[][]
    graph_edges: { from: number; to: number }[]
  }) =>
    fetchJson<{ metrics: Record<string, number> }>('/evaluate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}
