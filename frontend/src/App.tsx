import { useState } from 'react'
import { MuteIcon, VolumeIcon } from './components/Icons'
import { PlayView } from './components/PlayView'
import { SettingsMenu } from './components/SettingsMenu'
import { SimulationsTab } from './components/SimulationsTab'
import { useBots } from './hooks/useBots'
import { useSettings } from './lib/settings'

type Tab = 'play' | 'simulations'

export default function App() {
  const [tab, setTab] = useState<Tab>('play')
  const [settings, updateSettings] = useSettings()
  const bots = useBots()

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">♞</span>
          <span className="brand-name">Chess Lab</span>
        </div>
        <nav className="tabs">
          {(['play', 'simulations'] as Tab[]).map((t) => (
            <button key={t} className={`tab ${tab === t ? 'on' : ''}`} onClick={() => setTab(t)}>
              {t === 'play' ? 'Play' : 'Simulations'}
            </button>
          ))}
        </nav>
        <div className="topbar-actions">
          <button
            className="icon-btn"
            onClick={() => updateSettings({ sound: !settings.sound })}
            aria-label={settings.sound ? 'Mute' : 'Unmute'}
            title={settings.sound ? 'Mute' : 'Unmute'}
          >
            {settings.sound ? <VolumeIcon size={20} /> : <MuteIcon size={20} />}
          </button>
          <SettingsMenu settings={settings} onChange={updateSettings} />
        </div>
      </header>

      <main>
        {/* Keep the game mounted while browsing simulations. */}
        <div hidden={tab !== 'play'}>
          <PlayView settings={settings} bots={bots.available} botsError={bots.error} />
        </div>
        {tab === 'simulations' && <SimulationsTab bots={bots.available} />}
      </main>
    </div>
  )
}
