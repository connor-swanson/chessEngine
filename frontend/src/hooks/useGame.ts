import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api'
import { describeMove, fenAfter } from '../lib/chess'
import { playSound, type SoundKind } from '../lib/sound'
import type { GameSetup, GameState, Side } from '../types'

export interface Clock {
  white: number // ms left at the start of the current turn
  black: number
  /** when the side to move's clock started running (null = stopped) */
  runningSince: number | null
  /** clock values after each ply; index 0 is the start */
  history: { white: number; black: number }[]
}

const BOT_DELAY_MS = 150 // lets the human's move animate before the reply lands
const WATCH_DELAY_MS = 700 // bot vs bot pace

/** Apply the moves between two ply counts to the clock (or restore it after a takeback). */
function advanceClock(c: Clock, fromPlies: number, toPlies: number, over: boolean, incMs: number): Clock {
  const now = Date.now()
  const t = { white: c.white, black: c.black }
  let history = c.history
  if (toPlies > fromPlies) {
    for (let p = fromPlies; p < toPlies; p++) {
      const mover: Side = p % 2 === 0 ? 'white' : 'black'
      if (p === fromPlies && c.runningSince !== null) t[mover] -= now - c.runningSince
      if (p > 0) t[mover] += incMs
      history = [...history.slice(0, p + 1), { ...t }]
    }
  } else if (toPlies < fromPlies) {
    Object.assign(t, history[toPlies] ?? history[0])
    history = history.slice(0, toPlies + 1)
  }
  // The clock starts after White's first move.
  return { ...t, history, runningSince: over || toPlies === 0 ? null : now }
}

export function clockRemaining(c: Clock, side: Side, turn: Side, now: number): number {
  const base = c[side]
  return c.runningSince !== null && side === turn ? base - (now - c.runningSince) : base
}

export function playerOf(game: Pick<GameState, 'white_player' | 'black_player'>, side: Side) {
  return side === 'white' ? game.white_player : game.black_player
}

function soundFor(prev: GameState, next: GameState): SoundKind | null {
  const a = prev.move_history.length
  const b = next.move_history.length
  if (b === a + 1) {
    const m = describeMove(prev.fen, next.move_history[b - 1])
    if (!m) return 'move'
    if (m.check) return 'check'
    if (m.captured) return 'capture'
    if (m.castle) return 'castle'
    return 'move'
  }
  return b !== a ? 'move' : null
}

export interface GameController {
  game: GameState | null
  setup: GameSetup | null
  clock: Clock | null
  /** FEN shown while a human move is waiting for the server */
  pendingFen: string | null
  thinking: boolean
  paused: boolean
  hint: string | null
  hintLoading: boolean
  error: string | null
  takebackPlies: number
  newGame: (setup: GameSetup) => Promise<void>
  move: (uci: string) => Promise<void>
  takeback: () => Promise<void>
  resign: () => Promise<void>
  requestHint: () => Promise<void>
  setPaused: (p: boolean) => void
  dismissError: () => void
}

export function useGame(sound: boolean): GameController {
  const [game, setGame] = useState<GameState | null>(null)
  const [setup, setSetup] = useState<GameSetup | null>(null)
  const [clock, setClock] = useState<Clock | null>(null)
  const [pendingFen, setPendingFen] = useState<string | null>(null)
  const [thinking, setThinking] = useState(false)
  const [paused, setPaused] = useState(false)
  const [hint, setHint] = useState<string | null>(null)
  const [hintLoading, setHintLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Bumped whenever the game is reset, taken back or ended, so in-flight bot
  // replies for an old position are ignored.
  const gen = useRef(0)
  const gameRef = useRef<GameState | null>(null)
  const setupRef = useRef<GameSetup | null>(null)
  const soundRef = useRef(sound)
  useEffect(() => {
    soundRef.current = sound
  }, [sound])

  const apply = useCallback((next: GameState) => {
    const prev = gameRef.current
    gameRef.current = next
    setGame(next)
    setPendingFen(null)
    setHint(null)
    const sameGame = prev?.id === next.id
    const fromPlies = sameGame ? prev.move_history.length : 0
    const tc = setupRef.current?.timeControl
    setClock((c) => (c && tc ? advanceClock(c, fromPlies, next.move_history.length, next.is_game_over, tc.inc * 1000) : c))
    if (soundRef.current && sameGame) {
      const s = soundFor(prev, next)
      if (s) playSound(s)
      if (next.is_game_over && !prev.is_game_over) setTimeout(() => playSound('end'), 180)
    }
  }, [])

  const fail = useCallback((e: unknown) => {
    setError(e instanceof Error ? e.message : String(e))
  }, [])

  const newGame = useCallback(
    async (s: GameSetup) => {
      gen.current++
      setThinking(false)
      setPaused(false)
      setError(null)
      try {
        const g = await api.createGame(s.white, s.black)
        setupRef.current = s
        setSetup(s)
        const ms = s.timeControl ? s.timeControl.base * 60_000 : 0
        setClock(s.timeControl ? { white: ms, black: ms, runningSince: null, history: [{ white: ms, black: ms }] } : null)
        gameRef.current = null
        apply(g)
        if (soundRef.current) playSound('start')
      } catch (e) {
        fail(e)
      }
    },
    [apply, fail],
  )

  const move = useCallback(
    async (uci: string) => {
      const g = gameRef.current
      if (!g || g.is_game_over) return
      setPendingFen(fenAfter(g.fen, uci))
      try {
        apply(await api.move(g.id, uci))
      } catch (e) {
        setPendingFen(null)
        fail(e)
      }
    },
    [apply, fail],
  )

  // Bot moves: whenever it's a bot's turn, ask the backend for its move.
  useEffect(() => {
    if (!game || game.is_game_over || paused) return
    if (playerOf(game, game.turn) === 'human') return
    const myGen = gen.current
    const watching = game.white_player !== 'human' && game.black_player !== 'human'
    const timer = setTimeout(async () => {
      setThinking(true)
      try {
        const next = await api.botMove(game.id)
        if (myGen === gen.current) apply(next)
      } catch (e) {
        // 409: the position changed (takeback) while the bot was thinking.
        if (myGen === gen.current && !(e instanceof ApiError && e.status === 409)) fail(e)
      } finally {
        if (myGen === gen.current) setThinking(false)
      }
    }, watching ? WATCH_DELAY_MS : BOT_DELAY_MS)
    return () => clearTimeout(timer)
  }, [game, paused, apply, fail])

  const humans = game ? (['white', 'black'] as Side[]).filter((s) => playerOf(game, s) === 'human') : []
  let takebackPlies = 0
  if (game && game.move_history.length > 0 && humans.length > 0) {
    const n = game.move_history.length
    const offBoard = game.termination === 'resignation' || game.termination?.startsWith('timeout')
    const lastMover: Side = n % 2 === 1 ? 'white' : 'black'
    if (offBoard) takebackPlies = 0
    else if (humans.length === 2 || lastMover === humans[0]) takebackPlies = 1
    else takebackPlies = n >= 2 ? 2 : 0
  }

  const takeback = useCallback(async () => {
    const g = gameRef.current
    if (!g || takebackPlies === 0) return
    gen.current++
    setThinking(false)
    try {
      apply(await api.undo(g.id, takebackPlies))
    } catch (e) {
      fail(e)
    }
  }, [takebackPlies, apply, fail])

  const resign = useCallback(async () => {
    const g = gameRef.current
    if (!g || g.is_game_over) return
    const hs = (['white', 'black'] as Side[]).filter((s) => playerOf(g, s) === 'human')
    const side = hs.length === 1 ? hs[0] : g.turn
    gen.current++
    setThinking(false)
    try {
      apply(await api.resign(g.id, side))
    } catch (e) {
      fail(e)
    }
  }, [apply, fail])

  const requestHint = useCallback(async () => {
    const g = gameRef.current
    if (!g || g.is_game_over) return
    setHintLoading(true)
    try {
      const h = await api.hint(g.id)
      if (gameRef.current?.fen === g.fen) setHint(h.move_uci)
    } catch (e) {
      fail(e)
    } finally {
      setHintLoading(false)
    }
  }, [fail])

  // Flag fall: check the side to move's clock.
  useEffect(() => {
    if (!game || game.is_game_over || !clock || clock.runningSince === null) return
    const left = clockRemaining(clock, game.turn, game.turn, clock.runningSince)
    const myGen = gen.current
    const timer = setTimeout(async () => {
      if (myGen !== gen.current) return
      gen.current++
      setThinking(false)
      try {
        apply(await api.resign(game.id, game.turn, 'timeout'))
      } catch (e) {
        fail(e)
      }
    }, Math.max(0, left))
    return () => clearTimeout(timer)
  }, [game, clock, apply, fail])

  return {
    game,
    setup,
    clock,
    pendingFen,
    thinking,
    paused,
    hint,
    hintLoading,
    error,
    takebackPlies,
    newGame,
    move,
    takeback,
    resign,
    requestHint,
    setPaused,
    dismissError: () => setError(null),
  }
}
