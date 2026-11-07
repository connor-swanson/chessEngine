import { useCallback, useState } from 'react'

export type BoardTheme = 'green' | 'brown' | 'blue' | 'slate'

export const BOARD_THEMES: Record<BoardTheme, { light: string; dark: string; label: string }> = {
  green: { light: '#e9edcc', dark: '#6f9a55', label: 'Meadow' },
  brown: { light: '#f0d9b5', dark: '#b58863', label: 'Walnut' },
  blue: { light: '#dee3e6', dark: '#7d98b0', label: 'Ice' },
  slate: { light: '#c9ccd3', dark: '#6b7183', label: 'Slate' },
}

export interface Settings {
  boardTheme: BoardTheme
  sound: boolean
  showLegalMoves: boolean
  showCoordinates: boolean
  autoQueen: boolean
  animations: boolean
}

const DEFAULTS: Settings = {
  boardTheme: 'green',
  sound: true,
  showLegalMoves: true,
  showCoordinates: true,
  autoQueen: false,
  animations: true,
}

const KEY = 'chess-settings-v1'

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS
  } catch {
    return DEFAULTS
  }
}

export function useSettings(): [Settings, (patch: Partial<Settings>) => void] {
  const [settings, setSettings] = useState<Settings>(load)
  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((s) => {
      const next = { ...s, ...patch }
      try {
        localStorage.setItem(KEY, JSON.stringify(next))
      } catch {
        /* storage unavailable */
      }
      return next
    })
  }, [])
  return [settings, update]
}
