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

export interface BFSTaskSpecific {
  type: 'bfs'
  query_node: number[]
  answer_sequence: number[][]
}

export interface ShortestPathTaskSpecific {
  type: 'shortest_path'
  query_node: number[]
  answer_nodes: number[][]
}

export interface ConCompFactorTaskSpecific {
  type: 'concomp_factor'
  answer_nodes: number[][]
  components: number[][][]
}

export type TaskSpecific = DepoTaskSpecific | BFSTaskSpecific | ShortestPathTaskSpecific | ConCompFactorTaskSpecific | null

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

export interface TaskTab {
  tab_name: string
  task_name: string
  config_yaml: string
}

export interface TaskTabsResponse {
  full_config_yaml: string
  tabs: TaskTab[]
}
