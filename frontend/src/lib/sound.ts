// Small synthesized board sounds (no audio files needed).

export type SoundKind = 'move' | 'capture' | 'check' | 'castle' | 'end' | 'start'

let ctx: AudioContext | null = null

function audio(): AudioContext | null {
  try {
    ctx ??= new AudioContext()
    if (ctx.state === 'suspended') void ctx.resume()
    return ctx
  } catch {
    return null
  }
}

/** A short filtered noise burst: sounds like a piece landing on wood. */
function knock(ac: AudioContext, at: number, freq: number, gain: number) {
  const len = Math.floor(ac.sampleRate * 0.05)
  const buf = ac.createBuffer(1, len, ac.sampleRate)
  const data = buf.getChannelData(0)
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 3
  const src = ac.createBufferSource()
  src.buffer = buf
  const filter = ac.createBiquadFilter()
  filter.type = 'bandpass'
  filter.frequency.value = freq
  filter.Q.value = 1.2
  const g = ac.createGain()
  g.gain.value = gain
  src.connect(filter).connect(g).connect(ac.destination)
  src.start(at)
}

function tone(ac: AudioContext, at: number, freq: number, dur: number, gain: number) {
  const osc = ac.createOscillator()
  osc.type = 'sine'
  osc.frequency.value = freq
  const g = ac.createGain()
  g.gain.setValueAtTime(0, at)
  g.gain.linearRampToValueAtTime(gain, at + 0.01)
  g.gain.exponentialRampToValueAtTime(0.0001, at + dur)
  osc.connect(g).connect(ac.destination)
  osc.start(at)
  osc.stop(at + dur + 0.02)
}

export function playSound(kind: SoundKind) {
  const ac = audio()
  if (!ac) return
  const t = ac.currentTime + 0.005
  switch (kind) {
    case 'move':
      knock(ac, t, 1400, 1.6)
      break
    case 'capture':
      knock(ac, t, 900, 2.2)
      knock(ac, t + 0.045, 1600, 1.2)
      break
    case 'castle':
      knock(ac, t, 1400, 1.4)
      knock(ac, t + 0.09, 1200, 1.4)
      break
    case 'check':
      knock(ac, t, 1400, 1.4)
      tone(ac, t, 880, 0.18, 0.08)
      break
    case 'start':
      tone(ac, t, 523, 0.15, 0.06)
      tone(ac, t + 0.1, 784, 0.25, 0.06)
      break
    case 'end':
      tone(ac, t, 659, 0.2, 0.07)
      tone(ac, t + 0.15, 523, 0.35, 0.07)
      break
  }
}
