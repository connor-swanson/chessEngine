import { useEffect, useRef } from 'react'
import type { Ply } from '../lib/chess'
import { FirstIcon, LastIcon, NextIcon, PrevIcon } from './Icons'

interface Props {
  plies: Ply[]
  /** ply index being shown (number of moves played), plies.length = live */
  viewPly: number
  onSelect: (ply: number) => void
}

const PIECE_GLYPH: Record<string, string> = { K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘' }

/** SAN with a figurine for the moving piece, e.g. "Nf3" -> "♘f3". */
function figurine(san: string) {
  const g = PIECE_GLYPH[san[0]]
  return g ? (
    <>
      <span className="figurine">{g}</span>
      {san.slice(1)}
    </>
  ) : (
    san
  )
}

export function MoveList({ plies, viewPly, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const activeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const el = activeRef.current
    const box = scrollRef.current
    if (!box) return
    if (!el || viewPly === plies.length) box.scrollTop = box.scrollHeight
    else el.scrollIntoView({ block: 'nearest' })
  }, [viewPly, plies.length])

  const rows = []
  for (let i = 0; i < plies.length; i += 2) {
    rows.push(
      <div className="move-row" key={i}>
        <span className="move-num">{i / 2 + 1}.</span>
        {[i, i + 1].map((j) =>
          j < plies.length ? (
            <button
              key={j}
              ref={viewPly === j + 1 ? activeRef : undefined}
              className={`move ${viewPly === j + 1 ? 'active' : ''}`}
              onClick={() => onSelect(j + 1)}
            >
              {figurine(plies[j].san)}
            </button>
          ) : (
            <span key={j} />
          ),
        )}
      </div>,
    )
  }

  const last = plies.length
  return (
    <div className="move-list">
      <div className="moves-scroll" ref={scrollRef}>
        {rows.length ? rows : <p className="empty">No moves yet</p>}
      </div>
      <div className="move-nav">
        <button className="icon-btn" title="First move (↑)" onClick={() => onSelect(0)} disabled={viewPly === 0}>
          <FirstIcon />
        </button>
        <button className="icon-btn" title="Previous (←)" onClick={() => onSelect(viewPly - 1)} disabled={viewPly === 0}>
          <PrevIcon />
        </button>
        <button className="icon-btn" title="Next (→)" onClick={() => onSelect(viewPly + 1)} disabled={viewPly >= last}>
          <NextIcon />
        </button>
        <button className="icon-btn" title="Latest (↓)" onClick={() => onSelect(last)} disabled={viewPly >= last}>
          <LastIcon />
        </button>
      </div>
    </div>
  )
}
