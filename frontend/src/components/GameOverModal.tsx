import { botProfile } from '../lib/bots'
import type { GameState, Side } from '../types'
import { CloseIcon } from './Icons'
import { Avatar } from './PlayerBar'

interface Props {
  game: GameState
  /** the human's side in a game against a bot */
  you: Side | null
  onClose: () => void
  onRematch: () => void
  onNewGame: () => void
}

function headline(game: GameState, you: Side | null) {
  const winner: Side | null = game.result === '1-0' ? 'white' : game.result === '0-1' ? 'black' : null
  if (!winner) return { title: 'Draw', tone: 'draw' }
  if (you) return winner === you ? { title: 'You won!', tone: 'win' } : { title: 'You lost', tone: 'loss' }
  return { title: `${winner === 'white' ? 'White' : 'Black'} wins`, tone: 'neutral' }
}

function reason(game: GameState) {
  switch (game.termination) {
    case 'checkmate':
      return 'by checkmate'
    case 'resignation':
      return 'by resignation'
    case 'timeout':
      return 'on time'
    case null:
      return ''
    default:
      return `by ${game.termination}`
  }
}

function label(game: GameState, side: Side) {
  const name = side === 'white' ? game.white_player : game.black_player
  const twoHumans = game.white_player === 'human' && game.black_player === 'human'
  return twoHumans ? (side === 'white' ? 'White' : 'Black') : botProfile(name).label
}

export function GameOverModal({ game, you, onClose, onRematch, onNewGame }: Props) {
  const { title, tone } = headline(game, you)
  const score = game.result === '1/2-1/2' ? '½ – ½' : game.result?.replace('-', ' – ')
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal game-over ${tone}`} onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <button className="icon-btn modal-close" onClick={onClose} aria-label="Close">
          <CloseIcon />
        </button>
        <h2>{title}</h2>
        <p className="reason">{reason(game)}</p>
        <div className="versus">
          <div className="versus-player">
            <Avatar name={game.white_player} size={52} />
            <span>{label(game, 'white')}</span>
          </div>
          <div className="score">{score}</div>
          <div className="versus-player">
            <Avatar name={game.black_player} size={52} />
            <span>{label(game, 'black')}</span>
          </div>
        </div>
        <div className="modal-actions">
          <button className="primary-btn" onClick={onRematch}>
            Rematch
          </button>
          <button className="secondary-btn" onClick={onNewGame}>
            New game
          </button>
        </div>
        <button className="text-btn" onClick={onClose}>
          Review the game
        </button>
      </div>
    </div>
  )
}
