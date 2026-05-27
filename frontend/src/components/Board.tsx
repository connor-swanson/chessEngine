import { Chessboard } from 'react-chessboard'

interface DropArgs {
  piece: { isSparePiece: boolean; position: string; pieceType: string }
  sourceSquare: string
  targetSquare: string | null
}

interface BoardProps {
  fen: string
  legalMoves: string[]
  onMove: (moveUci: string) => void
  boardOrientation?: 'white' | 'black'
}

export function Board({ fen, legalMoves, onMove, boardOrientation = 'white' }: BoardProps) {
  const onPieceDrop = ({ piece, sourceSquare, targetSquare }: DropArgs): boolean => {
    if (!targetSquare) return false

    let moveUci = `${sourceSquare}${targetSquare}`

    // Auto-promote to queen for pawn promotions (pieceType is e.g. "wP" or "bP")
    const isPawn = piece.pieceType[1] === 'P'
    const isPromoRank = targetSquare[1] === '8' || targetSquare[1] === '1'
    if (isPawn && isPromoRank) moveUci += 'q'

    if (!legalMoves.some((m) => m.startsWith(moveUci.slice(0, 4)))) return false

    onMove(moveUci)
    return true
  }

  return (
    <div style={{ width: '480px', height: '480px', flexShrink: 0 }}>
      <Chessboard
        options={{
          position: fen,
          boardOrientation,
          onPieceDrop,
        }}
      />
    </div>
  )
}
