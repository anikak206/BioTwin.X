import { useEffect, useState } from 'react'
import { getPatient } from '@/services/api'
import { PatientReport } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Brain, Activity, FileText } from 'lucide-react'

export default function Insights() {
  const [report, setReport] = useState<PatientReport | null>(null)
  useEffect(() => {
    getPatient('BTX-0001').then(setReport).catch(() => {})
  }, [])
  if (!report) return <div className="text-muted">Loading insights...</div>
  return (
    <div className="space-y-8">
      <header>
        <p className="text-sm uppercase tracking-[0.4em] text-muted">AI Engine</p>
        <h1 className="text-4xl font-semibold tracking-tight">Actionable Insights</h1>
        <p className="text-muted">Evidence-driven recommendations with explainability</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Brain className="h-5 w-5" /> Primary Actions</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {report.actions?.map((a,i)=>(
              <div key={i} className="rounded-xl border border-border bg-panel/60 p-5">
                <div className="flex items-center justify-between"><span className="text-xs uppercase tracking-widest text-muted">{a.urgency}</span><span className="text-xs text-muted">{a.modality}</span></div>
                <h3 className="mt-2 text-lg font-medium">{a.title}</h3>
                <p className="mt-2 text-sm text-muted">{a.rationale}</p>
                <ul className="mt-3 space-y-1 text-sm text-muted">{a.evidence?.map((e,idx)=><li key={idx}>• {e}</li>)}</ul>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" /> Clinical Observations</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {report.insights?.map((ins,i)=>(
              <div key={i} className="rounded-xl border border-border bg-panel/60 p-4">
                <div className="flex items-center justify-between"><p className="font-medium">{ins.title}</p><p className="text-sm text-muted">{ins.score}/100</p></div>
                <p className="mt-2 text-sm text-muted">{ins.detail}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
