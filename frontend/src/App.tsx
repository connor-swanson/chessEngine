import { useState } from 'react'
import { Board } from './components/Board'
import { GameControls } from './components/GameControls'
import { SimulationsTab } from './components/SimulationsTab'
import { useGameSocket } from './hooks/useGameSocket'
import type { GameState } from './types'

const API = 'http://localhost:8000'

type Tab = 'play' | 'simulations'

export default function App() {
  const [tab, setTab] = useState<Tab>('play')
  const [gameId, setGameId] = useState<number | null>(null)
  const [localState, setLocalState] = useState<GameState | null>(null)
  const [boardOrientation, setBoardOrientation] = useState<'white' | 'black'>('white')
  const [whitePlayer, setWhitePlayer] = useState('human')
  const [blackPlayer, setBlackPlayer] = useState('random')

  const socketState = useGameSocket(gameId)
  const game = socketState ?? localState

  const runBotMoves = async (id: number, state: GameState, whiteP: string, blackP: string) => {
    let current = state
    while (!current.is_game_over) {
      if ((current.turn === 'white' ? whiteP : blackP) === 'human') break
      const res = await fetch(`${API}/games/${id}/bot-move`, { method: 'POST' })
      if (!res.ok) break
      current = await res.json()
      setLocalState(current)
    }
  }

  const newGame = async (white: string, black: string) => {
    setWhitePlayer(white)
    setBlackPlayer(black)
    const res = await fetch(`${API}/games`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ white_player: white, black_player: black }),
    })
    const state: GameState = await res.json()
    setGameId(state.id)
    setLocalState(state)
    await runBotMoves(state.id, state, white, black)
  }

  const makeMove = async (moveUci: string) => {
    if (!gameId || !game || game.is_game_over) return
    const res = await fetch(`${API}/games/${gameId}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ move_uci: moveUci }),
    })
    if (res.ok) {
      const newState: GameState = await res.json()
      setLocalState(newState)
      await runBotMoves(gameId, newState, whitePlayer, blackPlayer)
    }
  }

  const botMove = async () => {
    if (!gameId) return
    const res = await fetch(`${API}/games/${gameId}/bot-move`, { method: 'POST' })
    if (res.ok) {
      const newState: GameState = await res.json()
      setLocalState(newState)
      await runBotMoves(gameId, newState, whitePlayer, blackPlayer)
    }
  }

  return (
    <div>
      <nav style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', padding: '0 2rem' }}>
        {(['play', 'simulations'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: tab === t ? '2px solid #fff' : '2px solid transparent',
              color: tab === t ? '#fff' : '#888',
              cursor: 'pointer',
              fontSize: '0.95rem',
              fontWeight: tab === t ? 600 : 400,
              padding: '0.75rem 1.25rem',
              textTransform: 'capitalize',
              marginBottom: -1,
            }}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === 'play' && (
        <div style={{ display: 'flex', gap: '2rem', padding: '2rem', alignItems: 'flex-start' }}>
          <Board
            fen={game?.fen ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'}
            legalMoves={game?.legal_moves ?? []}
            onMove={makeMove}
            boardOrientation={boardOrientation}
          />
          <GameControls
            turn={game?.turn ?? 'white'}
            isGameOver={game?.is_game_over ?? false}
            result={game?.result ?? null}
            moveHistory={game?.move_history ?? []}
            onNewGame={newGame}
            onBotMove={botMove}
            onFlipBoard={() => setBoardOrientation((o) => (o === 'white' ? 'black' : 'white'))}
          />
        </div>
      )}

      {tab === 'simulations' && <SimulationsTab />}
    </div>
  )
}
