import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('glass relative overflow-hidden', className)}>{children}</div>
}

export function CardHeader({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('flex items-center justify-between px-6 pt-6', className)}>{children}</div>
}

export function CardTitle({ className, children }: { className?: string; children: ReactNode }) {
  return <h3 className={cn('text-lg font-medium tracking-tight', className)}>{children}</h3>
}

export function CardDescription({ className, children }: { className?: string; children: ReactNode }) {
  return <p className={cn('text-sm text-muted', className)}>{children}</p>
}

export function CardContent({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('px-6 pb-6 pt-4', className)}>{children}</div>
}
