import { useEffect, useState } from 'react'
import { getPatients } from '@/services/api'
import { Patient } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { TierBadge } from '@/components/ui/badges'
import { Activity, AlertTriangle, HeartPulse, Timer } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function Overview() {
  const [patients, setPatients] = useState<Patient[]>([])
  const nav = useNavigate()

  useEffect(() => {
    getPatients(25).then((res) => setPatients(res.patients || [])).catch(() => {})
  }, [])

  const critical = patients.filter((p) => p.tier === 'CRITICAL').length
  const high = patients.filter((p) => p.tier === 'HIGH').length
  const imminent = patients.filter((p) => (p.pred_ttx_h || 99) <= 12).length

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <p className="text-sm uppercase tracking-[0.4em] text-muted">Clinician Console</p>
        <h1 className="text-4xl font-semibold tracking-tight">BioTwin.X Triage Overview</h1>
        <p className="text-muted">Real-time prioritization of cardiometabolic risk across asynchronous streams</p>
      </header>

      <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
        {[{label: 'Monitored', value: patients.length, icon: UsersIcon}, {label: 'Critical', value: critical, icon: AlertTriangle}, {label: 'High Risk', value: high, icon: HeartPulse}, {label: 'Predicted < 12h', value: imminent, icon: Timer}].map((k) => (
          <Card key={k.label} className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted">{k.label}</p>
                <p className="mt-2 text-3xl font-semibold">{k.value}</p>
              </div>
              <k.icon className="h-10 w-10 text-accent/80" />
            </div>
          </Card>
        ))}
      </section>

      <Card>
        <CardHeader>
          <div>
            <CardTitle>Top Priority Patients</CardTitle>
            <p className="text-sm text-muted">Ranked by Clinical Risk Index</p>
          </div>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-3">Patient</th>
                <th className="pb-3">CRI</th>
                <th className="pb-3">Tier</th>
                <th className="pb-3">Dominant</th>
                <th className="pb-3">TTX (h)</th>
                <th className="pb-3">Urgency</th>
                <th className="pb-3">Headline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {patients.slice(0, 10).map((p) => (
                <tr key={p.patient_id} onClick={() => nav(`/patients/${p.patient_id}`)} className="cursor-pointer hover:bg-white/2">
                  <td className="py-4 font-medium">{p.patient_id}</td>
                  <td className="py-4">{Math.round(p.risk_index)}</td>
                  <td className="py-4"><TierBadge tier={p.tier} /></td>
                  <td className="py-4 capitalize">{(p.dominant_event || '').replace('_', ' ')}</td>
                  <td className="py-4">{p.pred_ttx_h?.toFixed(1)}</td>
                  <td className="py-4">{p.urgency}</td>
                  <td className="py-4 text-muted">{p.headline}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}

function UsersIcon(props: any) {
  return <Activity {...props} />
}
