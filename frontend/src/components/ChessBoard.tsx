import { useMemo, useState, type CSSProperties } from 'react'
import { Chessboard, defaultPieces, type PieceDropHandlerArgs, type SquareHandlerArgs } from 'react-chessboard'
import { checkedKingSquare } from '../lib/chess'
import { BOARD_THEMES, type Settings } from '../lib/settings'
import type { Side } from '../types'

interface Props {
  fen: string
  orientation: Side
  /** colour the user may move right now, or null when the board is read-only */
  movable: Side | null
  legalMoves: string[]
  lastMove: { from: string; to: string } | null
  hint: string | null
  settings: Settings
  onMove: (uci: string) => void
}

const PROMOTION_PIECES = ['q', 'r', 'b', 'n'] as const

export function ChessBoard({ fen, orientation, movable, legalMoves, lastMove, hint, settings, onMove }: Props) {
  const [selected, setSelected] = useState<string | null>(null)
  const [promotion, setPromotion] = useState<{ from: string; to: string } | null>(null)
  const [prevFen, setPrevFen] = useState(fen)
  const theme = BOARD_THEMES[settings.boardTheme]
  const moverPrefix = movable === 'white' ? 'w' : movable === 'black' ? 'b' : null

  // Drop any selection when the position changes.
  if (fen !== prevFen) {
    setPrevFen(fen)
    setSelected(null)
    setPromotion(null)
  }

  const targets = useMemo(
    () => (selected && movable ? legalMoves.filter((m) => m.startsWith(selected)).map((m) => m.slice(2, 4)) : []),
    [selected, movable, legalMoves],
  )

  const tryMove = (from: string, to: string): boolean => {
    const candidates = legalMoves.filter((m) => m.startsWith(from + to))
    if (candidates.length === 0) return false
    setSelected(null)
    if (candidates.length > 1) {
      // a promotion: one candidate per piece
      if (settings.autoQueen) onMove(from + to + 'q')
      else setPromotion({ from, to })
      return true
    }
    onMove(candidates[0])
    return true
  }

  const ownPiece = (pieceType: string | undefined) => !!moverPrefix && !!pieceType && pieceType[0] === moverPrefix

  const onSquareClick = ({ square, piece }: SquareHandlerArgs) => {
    if (!movable || promotion) return
    if (selected && targets.includes(square)) {
      tryMove(selected, square)
      return
    }
    if (ownPiece(piece?.pieceType) && square !== selected) setSelected(square)
    else setSelected(null)
  }

  const onPieceDrop = ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
    if (!targetSquare || targetSquare === sourceSquare) return false
    return tryMove(sourceSquare, targetSquare)
  }

  const squareStyles = useMemo(() => {
    const s: Record<string, CSSProperties> = {}
    const add = (sq: string, style: CSSProperties) => (s[sq] = { ...s[sq], ...style })
    if (lastMove) {
      add(lastMove.from, { backgroundColor: 'var(--hl-last)' })
      add(lastMove.to, { backgroundColor: 'var(--hl-last)' })
    }
    const king = checkedKingSquare(fen)
    if (king) {
      add(king, {
        background: 'radial-gradient(circle, rgba(255,40,40,.95) 0%, rgba(230,20,20,.6) 35%, rgba(200,0,0,0) 72%)',
      })
    }
    if (selected) add(selected, { backgroundColor: 'var(--hl-select)' })
    if (settings.showLegalMoves) {
      const occupied = new Set(boardSquares(fen))
      for (const t of targets) {
        add(t, {
          backgroundImage: occupied.has(t)
            ? 'radial-gradient(circle, transparent 0%, transparent 79%, var(--hl-dot) 80%)'
            : 'radial-gradient(circle, var(--hl-dot) 0%, var(--hl-dot) 17%, transparent 18%)',
        })
      }
    }
    return s
  }, [fen, lastMove, selected, targets, settings.showLegalMoves])

  const arrows = useMemo(
    () => (hint ? [{ startSquare: hint.slice(0, 2), endSquare: hint.slice(2, 4), color: 'rgba(80, 160, 255, 0.85)' }] : []),
    [hint],
  )

  const promoColor = moverPrefix ?? 'w'

  return (
    <div className="board-frame">
      <Chessboard
        options={{
          id: 'play-board',
          position: fen,
          boardOrientation: orientation,
          squareStyles,
          arrows,
          lightSquareStyle: { backgroundColor: theme.light },
          darkSquareStyle: { backgroundColor: theme.dark },
          lightSquareNotationStyle: { color: theme.dark, fontWeight: 600 },
          darkSquareNotationStyle: { color: theme.light, fontWeight: 600 },
          showNotation: settings.showCoordinates,
          showAnimations: settings.animations,
          animationDurationInMs: 180,
          allowDragging: !!movable && !promotion,
          canDragPiece: ({ piece }) => ownPiece(piece.pieceType),
          onPieceDrag: ({ square }) => square && setSelected(square),
          onPieceDrop,
          onSquareClick,
          onSquareRightClick: () => setSelected(null),
          dropSquareStyle: { boxShadow: 'inset 0 0 0 4px rgba(255,255,255,.55)' },
          boardStyle: { borderRadius: 6, overflow: 'hidden' },
        }}
      />
      {promotion && (
        <div className="promotion-overlay" onClick={() => setPromotion(null)}>
          <div className="promotion-picker" onClick={(e) => e.stopPropagation()}>
            <span className="promotion-title">Promote to</span>
            <div className="promotion-options">
              {PROMOTION_PIECES.map((p) => {
                const Piece = defaultPieces[`${promoColor}${p.toUpperCase()}`]
                return (
                  <button
                    key={p}
                    className="promotion-option"
                    onClick={() => {
                      onMove(promotion.from + promotion.to + p)
                      setPromotion(null)
                    }}
                    aria-label={`Promote to ${p}`}
                  >
                    <Piece />
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/** Occupied squares from a FEN, e.g. ["a8", "b8", ...]. */
function boardSquares(fen: string): string[] {
  const out: string[] = []
  fen
    .split(' ')[0]
    .split('/')
    .forEach((row, r) => {
      let f = 0
      for (const ch of row) {
        if (/\d/.test(ch)) f += Number(ch)
        else out.push('abcdefgh'[f++] + (8 - r))
      }
    })
  return out
}
