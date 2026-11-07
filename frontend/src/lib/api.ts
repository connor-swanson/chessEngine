import type { BotInfo, GameState, Side, SimulationStatus } from '../types'

export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    })
  } catch {
    throw new ApiError(0, `Can't reach the backend at ${API_URL}. Is it running?`)
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json() as Promise<T>
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  bots: () => request<{ bots: BotInfo[]; available: string[] }>('/bots'),
  createGame: (white: string, black: string) =>
    post<GameState>('/games', { white_player: white, black_player: black }),
  move: (id: number, uci: string) => post<GameState>(`/games/${id}/move`, { move_uci: uci }),
  botMove: (id: number) => post<GameState>(`/games/${id}/bot-move`),
  undo: (id: number, plies: number) => post<GameState>(`/games/${id}/undo`, { plies }),
  resign: (id: number, color: Side, reason: 'resignation' | 'timeout' = 'resignation') =>
    post<GameState>(`/games/${id}/resign`, { color, reason }),
  hint: (id: number) => request<{ move_uci: string; bot: string }>(`/games/${id}/hint`),
  startSimulation: (white: string, black: string, numGames: number) =>
    post<SimulationStatus>('/simulations', { num_games: numGames, white_player: white, black_player: black }),
  simulation: (id: number) => request<SimulationStatus>(`/simulations/${id}`),
}
