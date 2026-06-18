export interface RunInfo {
  id: string
  exp: string
  run: string
  config: Record<string, string>
}

export interface RunsResponse {
  runs: RunInfo[]
}

export interface SearchResponse {
  runs: RunInfo[]
}
