import { useState } from 'react'

const PLAYER_OPTIONS = ['human', 'random', 'heuristic', 'stockfish-easy', 'stockfish-medium', 'stockfish']

interface GameControlsProps {
  turn: string
  isGameOver: boolean
  result: string | null
  moveHistory: string[]
  onNewGame: (white: string, black: string) => void
  onBotMove: () => void
  onFlipBoard: () => void
}

export function GameControls({
  turn,
  isGameOver,
  result,
  moveHistory,
  onNewGame,
  onBotMove,
  onFlipBoard,
}: GameControlsProps) {
  const [white, setWhite] = useState('human')
  const [black, setBlack] = useState('random')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minWidth: 200 }}>
      <div>
        <h3 style={{ margin: '0 0 0.5rem' }}>New Game</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <label>
            White:{' '}
            <select value={white} onChange={(e) => setWhite(e.target.value)}>
              {PLAYER_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label>
            Black:{' '}
            <select value={black} onChange={(e) => setBlack(e.target.value)}>
              {PLAYER_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <button onClick={() => onNewGame(white, black)}>Start</button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button onClick={onBotMove} disabled={isGameOver}>Bot Move</button>
        <button onClick={onFlipBoard}>Flip Board</button>
      </div>

      <div>
        {isGameOver ? (
          <p style={{ color: 'green', fontWeight: 'bold' }}>Game over: {result}</p>
        ) : (
          <p>Turn: {turn}</p>
        )}
      </div>

      {moveHistory.length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 0.4rem' }}>Moves</h4>
          <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', maxHeight: 300, overflowY: 'auto' }}>
            {moveHistory.map((m, i) => (
              <span key={i} style={{ marginRight: '0.4rem' }}>
                {i % 2 === 0 ? `${Math.floor(i / 2) + 1}. ` : ''}{m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
