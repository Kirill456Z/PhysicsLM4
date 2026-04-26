export interface NodeData {
  tokens: number[]
}

export interface EdgeData {
  from: number
  to: number
}

export interface GraphData {
  nodes: NodeData[]
  edges: EdgeData[]
  n_nodes: number
}

export interface DepoTaskSpecific {
  type: 'depo'
  query_nodes: number[][]
  answer_nodes: number[][]
  num_hops: number[]
  answer_start_index: number
}

export type TaskSpecific = DepoTaskSpecific | null

export interface SampleResponse {
  graph: GraphData
  task_index: number
  task_name: string
  base_vocab_size: number
  context: number[]
  loss_mask: number[]
  context_padded: number[]
  labels: number[]
  task_specific: TaskSpecific
  answer_start_index: number
}

export interface BatchResponse {
  samples: SampleResponse[]
  count: number
}

export interface ValidateResponse {
  valid: boolean
  errors: string[]
  parsed?: { tasks: string[] }
}
