import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getPatient, getSeries } from '@/services/api'
import { PatientReport, SeriesPoint } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RiskGauge, TierBadge, UrgencyBadge } from '@/components/ui/badges'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, CartesianGrid } from 'recharts'

export default function PatientDetail() {
  const { id } = useParams()
  const [report, setReport] = useState<PatientReport | null>(null)
  const [series, setSeries] = useState<SeriesPoint[]>([])

  useEffect(() => {
    if (!id) return
    getPatient(id).then(setReport).catch(() => {})
    getSeries(id, 72).then((res) => setSeries(res.points || [])).catch(() => {})
  }, [id])

  if (!report) return <div className="text-muted">Loading patient...</div>

  const probs = report.event_probs || {}

  return (
    <div className="space-y-10">
      <header className="flex flex-wrap items-center justify-between gap-6">
        <div>
          <p className="text-sm uppercase tracking-[0.4em] text-muted">Patient Drilldown</p>
          <h1 className="text-4xl font-semibold tracking-tight">{report.patient_id}</h1>
          <p className="text-muted">As of {(report as any).as_of ? new Date((report as any).as_of).toLocaleString() : 'Now'}</p>
        </div>
        <div className="flex items-center gap-4">
          <TierBadge tier={report.tier} />
          <div className="text-right">
            <p className="text-sm text-muted">Backend</p>
            <p className="font-medium">{report.backend}</p>
          </div>
        </div>
      </header>

      <section className="grid gap-6 xl:grid-cols-[1fr_1.2fr_1fr]">
        <Card className="flex flex-col items-center justify-center p-8">
          <RiskGauge value={report.risk_index} />
          <p className="mt-4 text-sm text-muted">Clinical Risk Index</p>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>12-Hour Event Probabilities</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {[{key:'silent_ischemia', label:'Silent Ischemia'}, {key:'glycemic_crash', label:'Glycemic Crash'}, {key:'autonomic_arrhythmia', label:'Autonomic Arrhythmia'}].map((e) => {
              const v = (probs[e.key] || 0) * 100
              return (
                <div key={e.key}>
                  <div className="flex items-center justify-between text-sm">
                    <span>{e.label}</span>
                    <span className="text-muted">{v.toFixed(0)}%</span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-white/5">
                    <div className="h-2 rounded-full bg-accent" style={{ width: `${v}%` }} />
                  </div>
                </div>
              )
            })}
            <p className="pt-2 text-sm text-muted">Dominant: <span className="text-foreground capitalize">{(report.dominant_event||'').replace('_',' ')}</span> • Predicted TTX: {report.predicted_ttx_hours?.toFixed(1)}h</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Genomics & Vitals</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4 text-sm">
            <div><p className="text-muted">Nocturnal RMSSD</p><p className="text-lg">{report.nocturnal_rmssd?.toFixed(0)} ms</p></div>
            <div><p className="text-muted">Baseline RMSSD</p><p className="text-lg">{report.baseline_rmssd?.toFixed(0)} ms</p></div>
            <div><p className="text-muted">Confidence</p><p className="text-lg">{Math.round((report.confidence||0)*100)}%</p></div>
            <div><p className="text-muted">Coverage</p><p className="text-lg">{Math.round((report.coverage||0)*100)}%</p></div>
            <div><p className="text-muted">Latency</p><p className="text-lg">{report.latency_ms?.toFixed(1)} ms</p></div>
            <div><p className="text-muted">PRS (CVD)</p><p className="text-lg">{report.static?.prs_cvd_z?.toFixed(2)}σ</p></div>
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Risk Trajectory (72h)</CardTitle>
        </CardHeader>
        <CardContent className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series}>
              <defs>
                <linearGradient id="cri" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="ts" tick={{ fill: '#8fa1bd', fontSize: 12 }} />
              <YAxis tick={{ fill: '#8fa1bd', fontSize: 12 }} domain={[0, 100]} />
              <Tooltip contentStyle={{ background: '#151d33', border: '1px solid #243049', borderRadius: 12 }} />
              <Area type="monotone" dataKey="risk_index" stroke="#38bdf8" strokeWidth={2} fillOpacity={1} fill="url(#cri)" />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <section className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>AI Actionable Insights Engine</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {report.actions?.map((a, i) => (
              <div key={i} className="rounded-xl border border-border bg-panel/60 p-5">
                <div className="flex items-center justify-between">
                  <UrgencyBadge urgency={a.urgency} />
                  <span className="text-sm text-muted">{a.modality}</span>
                </div>
                <h4 className="mt-3 text-lg font-medium">{a.title}</h4>
                <p className="mt-2 text-sm text-muted">{a.rationale}</p>
                <ul className="mt-3 space-y-2 text-sm">
                  {a.evidence?.map((e, idx) => <li key={idx} className="text-muted">• {e}</li>)}
                </ul>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Supporting Observations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {report.insights?.map((ins, i) => (
              <div key={i} className="rounded-xl border border-border bg-panel/60 p-4">
                <div className="flex items-center justify-between">
                  <p className="font-medium">{ins.title}</p>
                  <span className="text-sm text-muted">{ins.score}/100</span>
                </div>
                <p className="mt-2 text-sm text-muted">{ins.detail}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>
    </div>
  )
}
