import { useEffect, useState } from 'react'

const API = 'http://localhost:8000'
const FALLBACK = ['random', 'heuristic']

// Bot names the backend reports as available (neural bots need torch + a checkpoint).
export function useBots(): string[] {
  const [bots, setBots] = useState<string[]>(FALLBACK)
  useEffect(() => {
    fetch(`${API}/bots`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d: { available: string[] }) => setBots(d.available))
      .catch(() => setBots(FALLBACK))
  }, [])
  return bots
}
