import { cn } from '@/lib/utils'
import { AlertCircle, ArrowDown, ArrowUp, HeartPulse } from 'lucide-react'

const tierColors: Record<string, string> = {
  LOW: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  ELEVATED: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  HIGH: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
  CRITICAL: 'bg-red-500/10 text-red-400 border-red-500/30',
}

const urgencyColors: Record<string, string> = {
  STAT: 'bg-red-500/15 text-red-400 border-red-500/40',
  URGENT: 'bg-orange-500/15 text-orange-400 border-orange-500/40',
  PRIORITY: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/40',
  ROUTINE: 'bg-sky-500/15 text-sky-400 border-sky-500/40',
  OPTIMISE: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40',
}

export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className={cn('inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border', tierColors[tier] || tierColors.LOW)}>
      {tier}
    </span>
  )
}

export function UrgencyBadge({ urgency }: { urgency: string }) {
  return (
    <span className={cn('inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border', urgencyColors[urgency] || urgencyColors.ROUTINE)}>
      {urgency}
    </span>
  )
}

export function RiskGauge({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value))
  const color = pct >= 75 ? '#ef4444' : pct >= 50 ? '#fb923c' : pct >= 25 ? '#f59e0b' : '#22c55e'
  const circumference = 2 * Math.PI * 42
  const dash = (pct / 100) * circumference
  return (
    <div className="relative h-48 w-48">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="42" stroke="rgba(148,163,184,0.2)" strokeWidth="10" fill="none" />
        <circle cx="50" cy="50" r="42" stroke={color} strokeWidth="10" fill="none" strokeDasharray={`${dash} ${circumference - dash}`} strokeLinecap="round" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-semibold tracking-tight">{Math.round(pct)}</span>
        <span className="text-xs uppercase tracking-[0.3em] text-muted">CRI</span>
      </div>
    </div>
  )
}
