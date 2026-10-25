import { useEffect, useState } from 'react'
import { getPatients } from '@/services/api'
import { Patient } from '@/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { TierBadge, UrgencyBadge } from '@/components/ui/badges'
import { useNavigate } from 'react-router-dom'

export default function Triage() {
  const [patients, setPatients] = useState<Patient[]>([])
  const nav = useNavigate()

  useEffect(() => {
    getPatients(100).then((res) => setPatients(res.patients || [])).catch(() => {})
  }, [])

  return (
    <div className="space-y-8">
      <header>
        <p className="text-sm uppercase tracking-[0.4em] text-muted">Triage</p>
        <h1 className="text-4xl font-semibold tracking-tight">Patient Board</h1>
        <p className="text-muted">Sorted by Clinical Risk Index • Evidence-driven prioritization</p>
      </header>
      <Card>
        <CardHeader>
          <CardTitle>All Patients</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-3">ID</th>
                <th className="pb-3">CRI</th>
                <th className="pb-3">Tier</th>
                <th className="pb-3">Ischemia</th>
                <th className="pb-3">Crash</th>
                <th className="pb-3">Arrhythmia</th>
                <th className="pb-3">TTX</th>
                <th className="pb-3">Urgency</th>
                <th className="pb-3">Coverage</th>
                <th className="pb-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {patients.map((p) => (
                <tr key={p.patient_id} onClick={() => nav(`/patients/${p.patient_id}`)} className="cursor-pointer hover:bg-white/2">
                  <td className="py-3 font-medium">{p.patient_id}</td>
                  <td className="py-3">{Math.round(p.risk_index)}</td>
                  <td className="py-3"><TierBadge tier={p.tier} /></td>
                  <td className="py-3">{Math.round((p.p_silent_ischemia||0)*100)}%</td>
                  <td className="py-3">{Math.round((p.p_glycemic_crash||0)*100)}%</td>
                  <td className="py-3">{Math.round((p.p_autonomic_arrhythmia||0)*100)}%</td>
                  <td className="py-3">{p.pred_ttx_h?.toFixed(1)}h</td>
                  <td className="py-3"><UrgencyBadge urgency={p.urgency} /></td>
                  <td className="py-3">{Math.round((p.coverage||0)*100)}%</td>
                  <td className="py-3 text-muted max-w-md truncate">{p.headline}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}
