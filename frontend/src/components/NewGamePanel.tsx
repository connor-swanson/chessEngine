import { useMemo, useState } from 'react'
import { botProfile, GROUP_LABEL, sortBots, type BotProfile } from '../lib/bots'
import type { GameSetup, TimeControl } from '../types'
import { Avatar } from './PlayerBar'

type Mode = 'bot' | 'watch' | 'local'
type ColorChoice = 'white' | 'random' | 'black'

const TIME_CONTROLS: { label: string; kind: string; tc: TimeControl | null }[] = [
  { label: '∞', kind: 'Unlimited', tc: null },
  { label: '3 + 2', kind: 'Blitz', tc: { base: 3, inc: 2 } },
  { label: '5 + 0', kind: 'Blitz', tc: { base: 5, inc: 0 } },
  { label: '10 + 0', kind: 'Rapid', tc: { base: 10, inc: 0 } },
  { label: '15 + 10', kind: 'Rapid', tc: { base: 15, inc: 10 } },
  { label: '30 + 0', kind: 'Classical', tc: { base: 30, inc: 0 } },
]

interface Props {
  bots: string[]
  botsError: string | null
  initial: GameSetup | null
  onStart: (setup: GameSetup) => void
  onCancel?: () => void
}

function pickDefault(bots: string[], prefs: string[]) {
  return prefs.find((b) => bots.includes(b)) ?? bots[0] ?? 'random'
}

export function NewGamePanel({ bots, botsError, initial, onStart, onCancel }: Props) {
  const [mode, setMode] = useState<Mode>('bot')
  const [showVariants, setShowVariants] = useState(false)
  const [opponent, setOpponent] = useState<string | null>(null)
  const [color, setColor] = useState<ColorChoice>('white')
  const [tcIndex, setTcIndex] = useState(0)
  const [watchWhite, setWatchWhite] = useState<string | null>(null)
  const [watchBlack, setWatchBlack] = useState<string | null>(null)

  const initialOpponent =
    initial && (initial.white === 'human') !== (initial.black === 'human')
      ? initial.white === 'human' ? initial.black : initial.white
      : null
  const chosen = opponent ?? (initialOpponent && bots.includes(initialOpponent) ? initialOpponent : pickDefault(bots, ['engine', 'ab2', 'ab1']))
  const ww = watchWhite ?? pickDefault(bots, ['engine', 'ab2'])
  const wb = watchBlack ?? pickDefault(bots, ['ab2', 'ab1'])

  const profiles = useMemo(() => sortBots(bots), [bots])
  const groups = useMemo(() => {
    const out = new Map<string, BotProfile[]>()
    for (const p of profiles) {
      if (p.group === 'variant' && !showVariants) continue
      const key = GROUP_LABEL[p.group]
      out.set(key, [...(out.get(key) ?? []), p])
    }
    return [...out.entries()]
  }, [profiles, showVariants])
  const variantCount = profiles.filter((p) => p.group === 'variant').length

  const tc = TIME_CONTROLS[tcIndex].tc
  const start = () => {
    if (mode === 'watch') return onStart({ white: ww, black: wb, timeControl: null })
    if (mode === 'local') return onStart({ white: 'human', black: 'human', timeControl: tc })
    const side = color === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : color
    onStart(side === 'white' ? { white: 'human', black: chosen, timeControl: tc } : { white: chosen, black: 'human', timeControl: tc })
  }
  const selected = botProfile(chosen)

  return (
    <div className="panel new-game">
      <div className="panel-header">
        <h2>New game</h2>
        {onCancel && (
          <button className="text-btn" onClick={onCancel}>
            Back to game
          </button>
        )}
      </div>

      <div className="segmented mode-tabs" role="tablist">
        {(
          [
            ['bot', 'Play a bot'],
            ['watch', 'Bot vs bot'],
            ['local', 'Two players'],
          ] as [Mode, string][]
        ).map(([m, label]) => (
          <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
            {label}
          </button>
        ))}
      </div>

      {botsError && <p className="notice warn">{botsError}</p>}

      <div className="panel-body">
        {mode === 'bot' && (
          <>
            <div className="bot-groups">
              {groups.map(([label, list]) => (
                <section key={label}>
                  <h3 className="section-label">{label}</h3>
                  <div className="bot-grid">
                    {list.map((p) => (
                      <button
                        key={p.name}
                        className={`bot-card ${p.name === chosen ? 'selected' : ''}`}
                        onClick={() => setOpponent(p.name)}
                        title={p.description}
                      >
                        <Avatar name={p.name} size={34} />
                        <span className="bot-card-text">
                          <span className="bot-card-name">{p.label}</span>
                          <span className="bot-card-level">{p.rating ? `≈${p.rating}` : p.level || p.name}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
              ))}
              {variantCount > 0 && (
                <button className="text-btn small" onClick={() => setShowVariants((v) => !v)}>
                  {showVariants ? 'Hide' : 'Show'} {variantCount} variants
                </button>
              )}
            </div>

            <div className="bot-detail">
              <Avatar name={selected.name} size={48} />
              <div>
                <div className="bot-detail-name">
                  {selected.label}
                  {selected.rating && <span className="rating">≈{selected.rating} Elo</span>}
                </div>
                <p>{selected.description}</p>
              </div>
            </div>

            <h3 className="section-label">I play as</h3>
            <div className="segmented color-choice">
              {(['white', 'random', 'black'] as ColorChoice[]).map((c) => (
                <button key={c} className={color === c ? 'on' : ''} onClick={() => setColor(c)} aria-label={c} title={c}>
                  <span className={`king-swatch ${c}`}>{c === 'random' ? '?' : '♚'}</span>
                  <span className="cap">{c}</span>
                </button>
              ))}
            </div>
          </>
        )}

        {mode === 'watch' && (
          <div className="watch-setup">
            {(
              [
                ['White', ww, setWatchWhite],
                ['Black', wb, setWatchBlack],
              ] as const
            ).map(([label, value, set]) => (
              <label key={label} className="field">
                <span className="section-label">{label}</span>
                <div className="select-with-avatar">
                  <Avatar name={value} size={30} />
                  <select value={value} onChange={(e) => set(e.target.value)}>
                    {profiles.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.label}
                        {p.rating ? ` (≈${p.rating})` : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </label>
            ))}
            <p className="muted">Moves play automatically. You can pause, step through and take over review at any time.</p>
          </div>
        )}

        {mode === 'local' && <p className="muted">Pass and play on one board. Flip the board between moves if you like.</p>}

        {mode !== 'watch' && (
          <>
            <h3 className="section-label">Time control</h3>
            <div className="tc-grid">
              {TIME_CONTROLS.map((t, i) => (
                <button key={t.label} className={`tc ${tcIndex === i ? 'on' : ''}`} onClick={() => setTcIndex(i)}>
                  <span className="tc-label">{t.label}</span>
                  <span className="tc-kind">{t.kind}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <button className="primary-btn big" onClick={start}>
        {mode === 'watch' ? 'Start watching' : 'Play'}
      </button>
    </div>
  )
}
