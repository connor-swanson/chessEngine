export interface GameState {
  id: number
  white_player: string
  black_player: string
  fen: string
  legal_moves: string[]
  turn: 'white' | 'black'
  result: string | null
  move_history: string[]
  is_game_over: boolean
}

export interface SimulationStatus {
  id: number
  status: 'running' | 'done'
  total: number
  completed: number
  white_wins: number
  black_wins: number
  draws: number
  white_player: string
  black_player: string
}
