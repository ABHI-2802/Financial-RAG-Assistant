export interface ConversationMessage {
  role: "user" | "assistant"
  content: string
}

export interface AskRequest {
  question: string
  top_k?: number
  source_filter?: string | null
  model?: string
  history?: ConversationMessage[]
}

export interface AskResponse {
  answer: string
  query: string
  citations: string[]
  retrieval_scores: number[]
  chunks_used: number
  model: string
  retrieval_ms: number
  generation_ms: number
  total_ms: number
  is_grounded: boolean
}

export interface HealthResponse {
  status: string
  model: string
  store_chunks: number
  store_path: string
}

export interface ModelsResponse {
  available: string[]
  current: string
}

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  response?: AskResponse
  isStreaming?: boolean
}

export type BankFilter = "all" | "interbank" | "scotiabank"
