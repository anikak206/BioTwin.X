export interface Patient {
  patient_id: string
  risk_index: number
  tier: 'LOW' | 'ELEVATED' | 'HIGH' | 'CRITICAL'
  p_silent_ischemia: number
  p_glycemic_crash: number
  p_autonomic_arrhythmia: number
  pred_ttx_h: number
  urgency: string
  headline: string
  dominant_event: string
  age: number
  sex: string
  nocturnal_rmssd: number
  rmssd_dev_baseline: number
  glu_min: number
  coverage: number
  latency_ms: number
  backend: string
}

export interface Action {
  urgency: 'STAT' | 'URGENT' | 'PRIORITY' | 'ROUTINE' | 'OPTIMISE'
  title: string
  rationale: string
  evidence: string[]
  modality: string
  biomarker: string
}

export interface Insight {
  title: string
  severity: string
  score: number
  detail: string
}

export interface PatientReport {
  patient_id: string
  risk_index: number
  tier: string
  event_probs: Record<string, number>
  dominant_event: string
  predicted_ttx_hours: number
  baseline_rmssd: number | null
  nocturnal_rmssd: number | null
  confidence: number
  coverage: number
  actions: Action[]
  insights: Insight[]
  context: Record<string, number>
  static: Record<string, any>
  explanation?: any
  latency_ms: number
  backend: string
}

export interface SeriesPoint {
  ts: string
  risk_index: number
  hrv_rmssd: number
  hr: number
  glucose: number
  lf_hf: number
}
