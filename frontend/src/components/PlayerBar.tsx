import { defaultPieces } from 'react-chessboard'
import { botProfile } from '../lib/bots'
import type { PieceSymbol } from '../lib/chess'
import type { Side } from '../types'

interface Props {
  player: string
  /** overrides the bot label, e.g. "White" in a two-player game */
  label?: string
  side: Side
  /** opponent pieces this player has captured */
  captured: PieceSymbol[]
  /** material lead in pawns (only shown when positive) */
  advantage: number
  clockMs: number | null
  clockRunning: boolean
  thinking: boolean
  toMove: boolean
  result?: 'win' | 'loss' | 'draw' | null
}

export function Avatar({ name, size = 40 }: { name: string; size?: number }) {
  const p = botProfile(name)
  return (
    <div
      className="avatar"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.5,
        background: `linear-gradient(135deg, hsl(${p.hue} 55% 46%), hsl(${p.hue + 25} 60% 32%))`,
      }}
    >
      {p.glyph}
    </div>
  )
}

function formatClock(ms: number): string {
  const t = Math.max(0, ms)
  const totalSec = Math.floor(t / 1000)
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  if (t < 10_000) return `${m}:${String(s).padStart(2, '0')}.${Math.floor((t % 1000) / 100)}`
  return `${m}:${String(s).padStart(2, '0')}`
}

export function PlayerBar({ player, label, side, captured, advantage, clockMs, clockRunning, thinking, toMove, result }: Props) {
  const p = botProfile(player)
  const pieceColor = side === 'white' ? 'b' : 'w' // captured pieces are the opponent's colour
  // Group identical pieces so they overlap like a stack.
  const groups: PieceSymbol[][] = []
  for (const piece of captured) {
    const last = groups[groups.length - 1]
    if (last && last[0] === piece) last.push(piece)
    else groups.push([piece])
  }
  return (
    <div className={`player-bar ${toMove ? 'to-move' : ''}`}>
      <Avatar name={player} />
      <div className="player-info">
        <div className="player-name">
          <span>{label ?? p.label}</span>
          {p.rating && <span className="rating">{p.rating}</span>}
          {player !== 'human' && <span className="badge">BOT</span>}
          {result && <span className={`result-chip ${result}`}>{result === 'win' ? '1' : result === 'loss' ? '0' : '½'}</span>}
          {thinking && (
            <span className="thinking" aria-label="thinking">
              <i />
              <i />
              <i />
            </span>
          )}
        </div>
        <div className="captured">
          {groups.map((g, i) => (
            <span key={i} className="captured-group">
              {g.map((piece, j) => {
                const Piece = defaultPieces[`${pieceColor}${piece.toUpperCase()}`]
                return (
                  <span key={j} className="captured-piece">
                    <Piece />
                  </span>
                )
              })}
            </span>
          ))}
          {advantage > 0 && <span className="advantage">+{advantage}</span>}
        </div>
      </div>
      {clockMs !== null && (
        <div className={`clock ${clockRunning ? 'running' : ''} ${clockMs < 20_000 ? 'low' : ''} side-${side}`}>
          {formatClock(clockMs)}
        </div>
      )}
    </div>
  )
}
