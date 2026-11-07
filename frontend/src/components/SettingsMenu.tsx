import { useEffect, useRef, useState } from 'react'
import { BOARD_THEMES, type BoardTheme, type Settings } from '../lib/settings'
import { GearIcon } from './Icons'

interface Props {
  settings: Settings
  onChange: (patch: Partial<Settings>) => void
}

const TOGGLES: [keyof Settings, string][] = [
  ['sound', 'Sound effects'],
  ['showLegalMoves', 'Show legal moves'],
  ['showCoordinates', 'Board coordinates'],
  ['animations', 'Piece animation'],
  ['autoQueen', 'Always promote to queen'],
]

export function SettingsMenu({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', esc)
    }
  }, [open])

  return (
    <div className="settings" ref={ref}>
      <button className={`icon-btn ${open ? 'on' : ''}`} onClick={() => setOpen((o) => !o)} aria-label="Settings" title="Settings">
        <GearIcon size={20} />
      </button>
      {open && (
        <div className="settings-menu">
          <h3 className="section-label">Board</h3>
          <div className="theme-row">
            {(Object.keys(BOARD_THEMES) as BoardTheme[]).map((t) => (
              <button
                key={t}
                className={`theme-swatch ${settings.boardTheme === t ? 'on' : ''}`}
                onClick={() => onChange({ boardTheme: t })}
                title={BOARD_THEMES[t].label}
                aria-label={BOARD_THEMES[t].label}
              >
                <span style={{ background: BOARD_THEMES[t].light }} />
                <span style={{ background: BOARD_THEMES[t].dark }} />
                <span style={{ background: BOARD_THEMES[t].dark }} />
                <span style={{ background: BOARD_THEMES[t].light }} />
              </button>
            ))}
          </div>
          <div className="toggles">
            {TOGGLES.map(([key, label]) => (
              <label key={key} className="toggle">
                <span>{label}</span>
                <input
                  type="checkbox"
                  checked={settings[key] as boolean}
                  onChange={(e) => onChange({ [key]: e.target.checked })}
                />
                <span className="switch" />
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
