import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { botProfile, sortBots } from '../lib/bots'
import type { SimulationStatus } from '../types'
import { Avatar } from './PlayerBar'

export function SimulationsTab({ bots }: { bots: string[] }) {
  const profiles = useMemo(() => sortBots(bots), [bots])
  const [white, setWhite] = useState('random')
  const [black, setBlack] = useState('random')
  const [numGames, setNumGames] = useState(10)
  const [simulations, setSimulations] = useState<SimulationStatus[]>([])
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const polls = useRef<Record<number, ReturnType<typeof setInterval>>>({})

  const startSimulation = async () => {
    setStarting(true)
    setError(null)
    try {
      const sim = await api.startSimulation(white, black, numGames)
      setSimulations((prev) => [sim, ...prev])
      if (sim.status !== 'done') {
        polls.current[sim.id] = setInterval(async () => {
          try {
            const updated = await api.simulation(sim.id)
            setSimulations((prev) => prev.map((s) => (s.id === updated.id ? updated : s)))
            if (updated.status === 'done') {
              clearInterval(polls.current[updated.id])
              delete polls.current[updated.id]
            }
          } catch {
            /* keep polling */
          }
        }, 1000)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  useEffect(() => {
    const p = polls.current
    return () => Object.values(p).forEach(clearInterval)
  }, [])

  const select = (value: string, set: (v: string) => void, label: string) => (
    <label className="field">
      <span className="section-label">{label}</span>
      <div className="select-with-avatar">
        <Avatar name={value} size={30} />
        <select value={value} onChange={(e) => set(e.target.value)}>
          {profiles.map((p) => (
            <option key={p.name} value={p.name}>
              {p.label}
              {p.rating ? ` (≈${p.rating})` : ''}
            </option>
          ))}
        </select>
      </div>
    </label>
  )

  return (
    <div className="sim-layout">
      <div className="panel sim-form">
        <div className="panel-header">
          <h2>Run simulations</h2>
        </div>
        <p className="muted">Play bots against each other in bulk. Games are saved to the database for training.</p>
        {select(white, setWhite, 'White')}
        {select(black, setBlack, 'Black')}
        <label className="field">
          <span className="section-label">Games</span>
          <input
            type="number"
            min={1}
            max={500}
            value={numGames}
            onChange={(e) => setNumGames(Math.max(1, Math.min(500, parseInt(e.target.value) || 1)))}
          />
        </label>
        {error && <p className="notice error">{error}</p>}
        <button className="primary-btn" onClick={() => void startSimulation()} disabled={starting}>
          {starting ? 'Starting…' : 'Run'}
        </button>
      </div>

      <div className="sim-results">
        {simulations.length === 0 && <div className="empty-state">Results appear here as games finish.</div>}
        {simulations.map((sim) => (
          <SimulationCard key={sim.id} sim={sim} />
        ))}
      </div>
    </div>
  )
}

function SimulationCard({ sim }: { sim: SimulationStatus }) {
  const pct = sim.total > 0 ? (sim.completed / sim.total) * 100 : 0
  const share = (n: number) => (sim.completed > 0 ? (n / sim.completed) * 100 : 0)
  const score = sim.completed > 0 ? ((sim.white_wins + sim.draws / 2) / sim.completed) * 100 : 0
  return (
    <div className="panel sim-card">
      <div className="sim-card-head">
        <div className="sim-players">
          <Avatar name={sim.white_player} size={28} />
          <span>{botProfile(sim.white_player).label}</span>
          <span className="vs">vs</span>
          <Avatar name={sim.black_player} size={28} />
          <span>{botProfile(sim.black_player).label}</span>
        </div>
        <span className={`status-pill ${sim.status === 'done' ? 'over' : 'thinking'}`}>
          {sim.status === 'done' ? 'Done' : `${sim.completed}/${sim.total}`}
        </span>
      </div>
      <div className="progress">
        <div style={{ width: `${pct}%` }} className={sim.status === 'done' ? 'done' : ''} />
      </div>
      {sim.completed > 0 && (
        <>
          <div className="wdl-bar">
            <div className="w" style={{ width: `${share(sim.white_wins)}%` }} title={`White wins: ${sim.white_wins}`} />
            <div className="d" style={{ width: `${share(sim.draws)}%` }} title={`Draws: ${sim.draws}`} />
            <div className="b" style={{ width: `${share(sim.black_wins)}%` }} title={`Black wins: ${sim.black_wins}`} />
          </div>
          <div className="wdl-legend">
            <span>
              <i className="w" /> White {sim.white_wins} ({share(sim.white_wins).toFixed(0)}%)
            </span>
            <span>
              <i className="d" /> Draw {sim.draws} ({share(sim.draws).toFixed(0)}%)
            </span>
            <span>
              <i className="b" /> Black {sim.black_wins} ({share(sim.black_wins).toFixed(0)}%)
            </span>
            <span className="muted">White score {score.toFixed(1)}%</span>
          </div>
        </>
      )}
    </div>
  )
}
