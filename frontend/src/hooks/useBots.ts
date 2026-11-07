import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const FALLBACK = ['random', 'heuristic']

export interface BotList {
  available: string[]
  loaded: boolean
  error: string | null
}

// Bot names the backend reports as available (neural bots need torch + a checkpoint).
export function useBots(): BotList {
  const [state, setState] = useState<BotList>({ available: FALLBACK, loaded: false, error: null })
  useEffect(() => {
    let alive = true
    api
      .bots()
      .then((d) => alive && setState({ available: d.available, loaded: true, error: null }))
      .catch((e: Error) => alive && setState({ available: FALLBACK, loaded: true, error: e.message }))
    return () => {
      alive = false
    }
  }, [])
  return state
}
