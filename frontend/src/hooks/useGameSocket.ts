import { useEffect, useRef, useState } from 'react'
import type { GameState } from '../types'

const WS_BASE = 'ws://localhost:8000'

export function useGameSocket(gameId: number | null) {
  const [socketState, setSocketState] = useState<GameState | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (gameId === null) return

    wsRef.current?.close()
    const ws = new WebSocket(`${WS_BASE}/games/${gameId}/ws`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      setSocketState(JSON.parse(event.data) as GameState)
    }

    return () => {
      ws.close()
    }
  }, [gameId])

  return socketState
}
