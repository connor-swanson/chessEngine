import { Chess, type PieceSymbol, type Square } from 'chess.js'

export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

export interface Ply {
  san: string
  uci: string
  from: string
  to: string
  /** FEN after this move */
  fen: string
  color: 'w' | 'b'
  captured?: PieceSymbol
  check: boolean
  castle: boolean
  promotion: boolean
}

function applyUci(chess: Chess, uci: string) {
  return chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] })
}

/** Replays a UCI move list into SAN + FEN per ply. */
export function replay(history: string[]): Ply[] {
  const chess = new Chess()
  const plies: Ply[] = []
  for (const uci of history) {
    const m = applyUci(chess, uci)
    plies.push({
      san: m.san,
      uci,
      from: m.from,
      to: m.to,
      fen: chess.fen(),
      color: m.color,
      captured: m.captured,
      check: m.san.includes('+') || m.san.includes('#'),
      castle: m.isKingsideCastle() || m.isQueensideCastle(),
      promotion: m.isPromotion(),
    })
  }
  return plies
}

/** Plays one UCI move on a FEN and returns the resulting FEN (for optimistic updates). */
export function fenAfter(fen: string, uci: string): string | null {
  try {
    const chess = new Chess(fen)
    applyUci(chess, uci)
    return chess.fen()
  } catch {
    return null
  }
}

export function describeMove(fen: string, uci: string): Omit<Ply, 'fen' | 'uci'> | null {
  try {
    const chess = new Chess(fen)
    const m = applyUci(chess, uci)
    return {
      san: m.san,
      from: m.from,
      to: m.to,
      color: m.color,
      captured: m.captured,
      check: chess.inCheck(),
      castle: m.isKingsideCastle() || m.isQueensideCastle(),
      promotion: m.isPromotion(),
    }
  } catch {
    return null
  }
}

export function checkedKingSquare(fen: string): string | null {
  const chess = new Chess(fen)
  if (!chess.inCheck()) return null
  return chess.findPiece({ type: 'k', color: chess.turn() })[0] ?? null
}

export const PIECE_VALUE: Record<PieceSymbol, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }
const START_COUNT: Record<PieceSymbol, number> = { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 }
const ORDER: PieceSymbol[] = ['p', 'n', 'b', 'r', 'q']

export interface Material {
  /** opponent pieces each side has taken, cheapest first */
  captured: { w: PieceSymbol[]; b: PieceSymbol[] }
  /** white material minus black material, in pawns */
  diff: number
}

/** Captured pieces and the material balance, read from a position. */
export function material(fen: string): Material {
  const count = { w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 } }
  for (const ch of fen.split(' ')[0]) {
    const lower = ch.toLowerCase() as PieceSymbol
    if (!(lower in PIECE_VALUE)) continue
    count[ch === lower ? 'b' : 'w'][lower]++
  }
  const missing = (color: 'w' | 'b') =>
    ORDER.flatMap((p) => Array<PieceSymbol>(Math.max(0, START_COUNT[p] - count[color][p])).fill(p))
  const total = (color: 'w' | 'b') => ORDER.reduce((s, p) => s + count[color][p] * PIECE_VALUE[p], 0)
  return { captured: { w: missing('b'), b: missing('w') }, diff: total('w') - total('b') }
}

export function pgn(history: string[], headers: Record<string, string>): string {
  const chess = new Chess()
  for (const uci of history) applyUci(chess, uci)
  for (const [k, v] of Object.entries(headers)) chess.setHeader(k, v)
  return chess.pgn()
}

export type { PieceSymbol, Square }
