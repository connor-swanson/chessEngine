export type BotGroup = 'engine' | 'classic' | 'stockfish' | 'variant'

export interface BotProfile {
  name: string
  label: string
  /** Elo on the project's Stockfish-anchored scale, only where it was measured */
  rating?: number
  level: string
  description: string
  group: BotGroup
  /** avatar colour */
  hue: number
  glyph: string
}

// Ratings come from train/results/*.json and the bot ladder in the README
// (anchored to Stockfish UCI_Elo 1320 at 0.1 s/move, roughly ±100).
const PROFILES: BotProfile[] = [
  { name: 'random', label: 'Randy', level: 'Beginner', glyph: '?', hue: 30, group: 'classic',
    description: 'Plays any legal move at random.' },
  { name: 'heuristic', label: 'Grabby', level: 'Novice', glyph: '♟', hue: 50, group: 'classic',
    description: 'Looks one move ahead and grabs material.' },
  { name: 'ab1', label: 'Alpha 1', rating: 1200, level: 'Casual', glyph: '♞', hue: 160, group: 'classic',
    description: 'Alpha-beta search, 1 ply plus captures.' },
  { name: 'ab2', label: 'Alpha 2', rating: 1470, level: 'Intermediate', glyph: '♝', hue: 200, group: 'classic',
    description: 'Alpha-beta search, 2 plies. The teacher the network imitated.' },
  { name: 'ab3', label: 'Alpha 3', rating: 1670, level: 'Advanced', glyph: '♜', hue: 260, group: 'classic',
    description: 'Alpha-beta search, 3 plies.' },
  { name: 'engine', label: 'Engine', rating: 1480, level: 'Intermediate', glyph: '◆', hue: 145, group: 'engine',
    description: 'The trained network with 200-simulation MCTS. The final result of the notebooks.' },
  { name: 'engine-fast', label: 'Engine Blitz', level: 'Casual', glyph: '◇', hue: 120, group: 'engine',
    description: 'Same network, 50 simulations. Quicker and a little weaker.' },
  { name: 'engine-policy', label: 'Engine Instinct', level: 'Novice', glyph: '○', hue: 95, group: 'engine',
    description: 'The raw network policy with no search at all.' },
  { name: 'mcts', label: 'AlphaZero net', level: 'Intermediate', glyph: '◈', hue: 175, group: 'engine',
    description: 'Best self-play network from notebook 4, 100 simulations.' },
  { name: 'sft', label: 'Imitator', level: 'Casual', glyph: '◎', hue: 75, group: 'engine',
    description: 'The supervised network from notebook 1, 100 simulations.' },
  { name: 'stockfish-easy', label: 'Stockfish 5', level: 'Strong', glyph: '♛', hue: 0, group: 'stockfish',
    description: 'Stockfish at skill level 5.' },
  { name: 'stockfish-medium', label: 'Stockfish 12', level: 'Expert', glyph: '♛', hue: 340, group: 'stockfish',
    description: 'Stockfish at skill level 12.' },
  { name: 'stockfish', label: 'Stockfish 20', level: 'Master', glyph: '♚', hue: 320, group: 'stockfish',
    description: 'Full-strength Stockfish at 0.1 s per move.' },
]

const BY_NAME = new Map(PROFILES.map((p) => [p.name, p]))

export const GROUP_LABEL: Record<BotGroup, string> = {
  engine: 'Trained engine',
  classic: 'Scripted bots',
  stockfish: 'Stockfish',
  variant: 'Variants',
}

export function botProfile(name: string): BotProfile {
  const known = BY_NAME.get(name)
  if (known) return known
  if (name === 'human') {
    return { name, label: 'You', level: '', glyph: '☺', hue: 220, group: 'variant', description: '' }
  }
  // epsilon variants such as heuristic-e20 or ab2-e10
  const m = name.match(/^(.*)-e(\d+)$/)
  const base = m ? BY_NAME.get(m[1]) : undefined
  return {
    name,
    label: base ? `${base.label} (${m![2]}% random)` : name,
    level: base?.level ?? '',
    glyph: base?.glyph ?? '•',
    hue: base?.hue ?? 210,
    group: 'variant',
    description: base ? `${base.label}, but plays a random move ${m![2]}% of the time.` : '',
  }
}

/** Display order: engines, scripted bots by strength, Stockfish, then variants. */
export function sortBots(names: string[]): BotProfile[] {
  const order = (p: BotProfile) => {
    const i = PROFILES.findIndex((q) => q.name === p.name)
    return i === -1 ? 1000 : i
  }
  const groupRank: Record<BotGroup, number> = { engine: 0, classic: 1, stockfish: 2, variant: 3 }
  return names
    .map(botProfile)
    .sort((a, b) => groupRank[a.group] - groupRank[b.group] || order(a) - order(b) || a.name.localeCompare(b.name))
}
