import type { RunsResponse, SearchResponse } from '../types'
// RunsResponse kept for the /api/runs debug endpoint

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
  getRuns: () => fetchJson<RunsResponse>('/runs'),

  search: (yamlFilter: string, minor: number) =>
    fetchJson<SearchResponse>('/search', {
      method: 'POST',
      body: JSON.stringify({ yaml_filter: yamlFilter, minor }),
    }),

}
