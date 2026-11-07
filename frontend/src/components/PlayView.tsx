import { useEffect, useMemo, useState } from 'react'
import { clockRemaining, playerOf, useGame } from '../hooks/useGame'
import { botProfile } from '../lib/bots'
import { material, pgn, replay, START_FEN } from '../lib/chess'
import type { Settings } from '../lib/settings'
import type { GameSetup, Side } from '../types'
import { ChessBoard } from './ChessBoard'
import { GameOverModal } from './GameOverModal'
import { BulbIcon, CopyIcon, DownloadIcon, FlagIcon, FlipIcon, PauseIcon, PlayIcon, PlusIcon, UndoIcon } from './Icons'
import { MoveList } from './MoveList'
import { NewGamePanel } from './NewGamePanel'
import { PlayerBar } from './PlayerBar'

interface Props {
  settings: Settings
  bots: string[]
  botsError: string | null
}

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(Date.now()), 100)
    return () => clearInterval(id)
  }, [active])
  return now
}

export function PlayView({ settings, bots, botsError }: Props) {
  const g = useGame(settings.sound)
  const { game, setup, clock } = g
  const [orientation, setOrientation] = useState<Side>('white')
  const [viewPly, setViewPly] = useState<number | null>(null) // null = live
  const [showSetup, setShowSetup] = useState(true)
  const [dismissedEnd, setDismissedEnd] = useState<string | null>(null)
  const [confirmResign, setConfirmResign] = useState(false)
  const [toast, setToast] = useState<string | null>(null)

  const plies = useMemo(() => replay(game?.move_history ?? []), [game?.move_history])
  const livePly = plies.length
  const shownPly = viewPly === null ? livePly : Math.min(viewPly, livePly)
  const reviewing = shownPly !== livePly
  const fen = reviewing ? (shownPly === 0 ? START_FEN : plies[shownPly - 1].fen) : g.pendingFen ?? game?.fen ?? START_FEN
  const lastPly = shownPly > 0 ? plies[shownPly - 1] : null
  const mat = material(fen)

  const humans = game ? (['white', 'black'] as Side[]).filter((s) => playerOf(game, s) === 'human') : []
  const you: Side | null = humans.length === 1 ? humans[0] : null
  const watching = !!game && humans.length === 0
  const humanToMove = !!game && !game.is_game_over && playerOf(game, game.turn) === 'human'
  const movable = humanToMove && !reviewing && !g.pendingFen ? game!.turn : null

  const clockActive = !!clock && clock.runningSince !== null && !!game && !game.is_game_over
  const now = useNow(clockActive)
  const clockFor = (side: Side) => (clock && game ? clockRemaining(clock, side, game.turn, clockActive ? now : 0) : null)

  const start = async (s: GameSetup) => {
    await g.newGame(s)
    setOrientation(s.black === 'human' && s.white !== 'human' ? 'black' : 'white')
    setViewPly(null)
    setShowSetup(false)
    setConfirmResign(false)
  }

  const rematch = () => {
    if (!setup) return
    const vsBot = (setup.white === 'human') !== (setup.black === 'human')
    void start(vsBot ? { ...setup, white: setup.black, black: setup.white } : setup)
  }

  // Keyboard: ←/→ step through moves, ↑/↓ jump to start/end, f flips.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
      if (!game) return
      const go = (p: number) => setViewPly(Math.max(0, Math.min(livePly, p)) === livePly ? null : Math.max(0, p))
      if (e.key === 'ArrowLeft') go(shownPly - 1)
      else if (e.key === 'ArrowRight') go(shownPly + 1)
      else if (e.key === 'ArrowUp') go(0)
      else if (e.key === 'ArrowDown') go(livePly)
      else if (e.key === 'f') setOrientation((o) => (o === 'white' ? 'black' : 'white'))
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [game, shownPly, livePly])

  useEffect(() => {
    if (!confirmResign) return
    const t = setTimeout(() => setConfirmResign(false), 3000)
    return () => clearTimeout(t)
  }, [confirmResign])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 1600)
    return () => clearTimeout(t)
  }, [toast])

  const pgnText = () =>
    game
      ? pgn(game.move_history, {
          Event: 'Casual game',
          Date: new Date().toISOString().slice(0, 10).replace(/-/g, '.'),
          White: botProfile(game.white_player).label,
          Black: botProfile(game.black_player).label,
          Result: game.result ?? '*',
          ...(game.termination ? { Termination: game.termination } : {}),
        })
      : ''

  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setToast(`${what} copied`)
    } catch {
      setToast('Copy failed')
    }
  }

  const download = () => {
    const blob = new Blob([pgnText()], { type: 'application/x-chess-pgn' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `game-${game?.id ?? 'new'}.pgn`
    a.click()
    URL.revokeObjectURL(url)
  }

  const endKey = game?.is_game_over ? `${game.id}:${livePly}:${game.result}` : null
  const showEnd = !!endKey && endKey !== dismissedEnd && !showSetup

  const top: Side = orientation === 'white' ? 'black' : 'white'
  const bottom: Side = orientation
  const resultFor = (side: Side) => {
    if (!game?.is_game_over) return null
    if (game.result === '1/2-1/2') return 'draw' as const
    return (game.result === '1-0') === (side === 'white') ? ('win' as const) : ('loss' as const)
  }
  const bar = (side: Side) =>
    game ? (
      <PlayerBar
        player={playerOf(game, side)}
        label={humans.length === 2 ? (side === 'white' ? 'White' : 'Black') : undefined}
        side={side}
        captured={mat.captured[side === 'white' ? 'w' : 'b']}
        advantage={side === 'white' ? mat.diff : -mat.diff}
        clockMs={clockFor(side)}
        clockRunning={clockActive && game.turn === side}
        thinking={g.thinking && game.turn === side}
        toMove={!game.is_game_over && game.turn === side}
        result={resultFor(side)}
      />
    ) : (
      <PlayerBar
        player={side === bottom ? 'human' : 'engine'}
        side={side}
        captured={[]}
        advantage={0}
        clockMs={null}
        clockRunning={false}
        thinking={false}
        toMove={false}
      />
    )

  let status: string
  if (!game) status = 'Pick an opponent to start'
  else if (game.is_game_over) status = 'Game over'
  else if (g.paused) status = 'Paused'
  else if (g.thinking) status = `${botProfile(playerOf(game, game.turn)).label} is thinking…`
  else if (humanToMove && you) status = 'Your move'
  else status = `${game.turn === 'white' ? 'White' : 'Black'} to move`

  const tcLabel = setup?.timeControl ? `${setup.timeControl.base} + ${setup.timeControl.inc}` : 'Unlimited'

  return (
    <div className="play-layout">
      <div className="board-column">
        {bar(top)}
        <div className="board-wrap">
          <ChessBoard
            fen={fen}
            orientation={orientation}
            movable={movable}
            legalMoves={game?.legal_moves ?? []}
            lastMove={lastPly ? { from: lastPly.from, to: lastPly.to } : null}
            hint={reviewing ? null : g.hint}
            settings={settings}
            onMove={(uci) => void g.move(uci)}
          />
          {reviewing && (
            <button className="review-banner" onClick={() => setViewPly(null)}>
              Viewing move {Math.ceil(shownPly / 2) || 'start'} · back to live game
            </button>
          )}
          {showEnd && game && (
            <GameOverModal
              game={game}
              you={you}
              onClose={() => setDismissedEnd(endKey)}
              onRematch={rematch}
              onNewGame={() => setShowSetup(true)}
            />
          )}
        </div>
        {bar(bottom)}
      </div>

      <aside className="side-column">
        {showSetup || !game ? (
          <NewGamePanel
            bots={bots}
            botsError={botsError}
            initial={setup}
            onStart={(s) => void start(s)}
            onCancel={game ? () => setShowSetup(false) : undefined}
          />
        ) : (
          <div className="panel game-panel">
            <div className="panel-header">
              <div>
                <h2>
                  {humans.length === 2 ? (
                    'Two players'
                  ) : (
                    <>
                      {botProfile(game.white_player).label} <span className="vs">vs</span> {botProfile(game.black_player).label}
                    </>
                  )}
                </h2>
                <span className="muted small">{tcLabel}</span>
              </div>
              <span className={`status-pill ${game.is_game_over ? 'over' : g.thinking ? 'thinking' : humanToMove ? 'your-turn' : ''}`}>
                {status}
              </span>
            </div>

            {g.error && (
              <div className="notice error" role="alert">
                <span>{g.error}</span>
                <button className="text-btn small" onClick={g.dismissError}>
                  Dismiss
                </button>
              </div>
            )}

            <MoveList plies={plies} viewPly={shownPly} onSelect={(p) => setViewPly(p >= livePly ? null : Math.max(0, p))} />

            {game.is_game_over && (
              <div className="result-line">
                <strong>{game.result === '1/2-1/2' ? '½–½' : game.result}</strong>
                <span>{game.termination}</span>
              </div>
            )}

            <div className="actions">
              {watching ? (
                <button className="action" onClick={() => g.setPaused(!g.paused)} disabled={game.is_game_over}>
                  {g.paused ? <PlayIcon /> : <PauseIcon />}
                  <span>{g.paused ? 'Resume' : 'Pause'}</span>
                </button>
              ) : (
                <>
                  <button className="action" onClick={() => void g.takeback()} disabled={g.takebackPlies === 0} title="Take back your last move">
                    <UndoIcon />
                    <span>Takeback</span>
                  </button>
                  <button className="action" onClick={() => void g.requestHint()} disabled={!humanToMove || g.hintLoading || reviewing} title="Show a suggested move">
                    <BulbIcon />
                    <span>{g.hintLoading ? 'Thinking…' : 'Hint'}</span>
                  </button>
                </>
              )}
              <button className="action" onClick={() => setOrientation((o) => (o === 'white' ? 'black' : 'white'))} title="Flip board (f)">
                <FlipIcon />
                <span>Flip</span>
              </button>
              {!watching && (
                <button
                  className={`action ${confirmResign ? 'danger' : ''}`}
                  disabled={game.is_game_over}
                  onClick={() => {
                    if (confirmResign) {
                      setConfirmResign(false)
                      void g.resign()
                    } else setConfirmResign(true)
                  }}
                  title="Resign"
                >
                  <FlagIcon />
                  <span>{confirmResign ? 'Confirm' : 'Resign'}</span>
                </button>
              )}
            </div>

            <button className="primary-btn" onClick={() => setShowSetup(true)}>
              <PlusIcon /> New game
            </button>

            <div className="export-row">
              <button className="text-btn small" onClick={() => void copy(pgnText(), 'PGN')}>
                <CopyIcon size={14} /> PGN
              </button>
              <button className="text-btn small" onClick={download}>
                <DownloadIcon size={14} /> Download
              </button>
              <button className="text-btn small" onClick={() => void copy(fen, 'FEN')}>
                <CopyIcon size={14} /> FEN
              </button>
              {toast && <span className="toast">{toast}</span>}
            </div>
          </div>
        )}
      </aside>
    </div>
  )
}
