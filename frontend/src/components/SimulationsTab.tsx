import { useState, useEffect, useRef } from 'react'
import type { SimulationStatus } from '../types'

const API = 'http://localhost:8000'
const BOT_OPTIONS = ['random', 'heuristic']

export function SimulationsTab() {
  const [white, setWhite] = useState('random')
  const [black, setBlack] = useState('random')
  const [numGames, setNumGames] = useState(10)
  const [simulations, setSimulations] = useState<SimulationStatus[]>([])
  const [running, setRunning] = useState(false)
  const pollRefs = useRef<Record<number, ReturnType<typeof setInterval>>>({})

  const startSimulation = async () => {
    setRunning(true)
    const res = await fetch(`${API}/simulations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_games: numGames, white_player: white, black_player: black }),
    })
    const sim: SimulationStatus = await res.json()
    setSimulations((prev) => [sim, ...prev])
    setRunning(false)

    if (sim.status !== 'done') {
      pollRefs.current[sim.id] = setInterval(async () => {
        const r = await fetch(`${API}/simulations/${sim.id}`)
        const updated: SimulationStatus = await r.json()
        setSimulations((prev) => prev.map((s) => (s.id === updated.id ? updated : s)))
        if (updated.status === 'done') {
          clearInterval(pollRefs.current[updated.id])
          delete pollRefs.current[updated.id]
        }
      }, 1000)
    }
  }

  useEffect(() => {
    return () => {
      Object.values(pollRefs.current).forEach(clearInterval)
    }
  }, [])

  return (
    <div style={{ padding: '2rem', maxWidth: 700 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginBottom: '2rem', maxWidth: 300 }}>
        <h3 style={{ margin: '0 0 0.4rem' }}>Run Simulations</h3>
        <label>
          White:{' '}
          <select value={white} onChange={(e) => setWhite(e.target.value)}>
            {BOT_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>
          Black:{' '}
          <select value={black} onChange={(e) => setBlack(e.target.value)}>
            {BOT_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>
          Games:{' '}
          <input
            type="number"
            min={1}
            max={500}
            value={numGames}
            onChange={(e) => setNumGames(Math.max(1, parseInt(e.target.value) || 1))}
            style={{ width: 70 }}
          />
        </label>
        <button onClick={startSimulation} disabled={running} style={{ marginTop: '0.4rem' }}>
          {running ? 'Starting…' : 'Run'}
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {simulations.map((sim) => (
          <SimulationCard key={sim.id} sim={sim} />
        ))}
      </div>
    </div>
  )
}

function SimulationCard({ sim }: { sim: SimulationStatus }) {
  const pct = sim.total > 0 ? (sim.completed / sim.total) * 100 : 0
  const whitePct = sim.completed > 0 ? (sim.white_wins / sim.completed) * 100 : 0
  const blackPct = sim.completed > 0 ? (sim.black_wins / sim.completed) * 100 : 0
  const drawPct = sim.completed > 0 ? (sim.draws / sim.completed) * 100 : 0

  return (
    <div style={{
      border: '1px solid #444',
      borderRadius: 6,
      padding: '1rem',
      background: '#1a1a1a',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <span style={{ fontWeight: 'bold' }}>
          Sim #{sim.id} — {sim.white_player} vs {sim.black_player}
        </span>
        <span style={{
          fontSize: '0.8rem',
          color: sim.status === 'done' ? '#4caf50' : '#ffb74d',
          fontWeight: 'bold',
          textTransform: 'uppercase',
        }}>
          {sim.status === 'done' ? 'Done' : `Running ${sim.completed}/${sim.total}`}
        </span>
      </div>

      <div style={{ background: '#333', borderRadius: 4, height: 8, marginBottom: '0.75rem', overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: sim.status === 'done' ? '#4caf50' : '#2196f3',
          transition: 'width 0.4s ease',
        }} />
      </div>

      {sim.completed > 0 && (
        <>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.4rem', overflow: 'hidden', borderRadius: 4, height: 20 }}>
            <div style={{ width: `${whitePct}%`, background: '#e0e0e0', transition: 'width 0.4s ease' }} title={`White wins: ${sim.white_wins}`} />
            <div style={{ width: `${drawPct}%`, background: '#9e9e9e', transition: 'width 0.4s ease' }} title={`Draws: ${sim.draws}`} />
            <div style={{ width: `${blackPct}%`, background: '#555', transition: 'width 0.4s ease' }} title={`Black wins: ${sim.black_wins}`} />
          </div>
          <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.85rem', color: '#bbb' }}>
            <span>W: {sim.white_wins} ({whitePct.toFixed(1)}%)</span>
            <span>D: {sim.draws} ({drawPct.toFixed(1)}%)</span>
            <span>B: {sim.black_wins} ({blackPct.toFixed(1)}%)</span>
          </div>
        </>
      )}
    </div>
  )
}
