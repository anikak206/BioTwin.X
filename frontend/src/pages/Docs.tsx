import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function Docs() {
  return (
    <div className="space-y-8">
      <header>
        <p className="text-sm uppercase tracking-[0.4em] text-muted">Documentation</p>
        <h1 className="text-4xl font-semibold tracking-tight">Model Card & Architecture</h1>
        <p className="text-muted">BioTwin.X — Cross-Modal Contrastive Digital Twin</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Technical Stack</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-muted">
            <p>Frontend: React + Vite + TypeScript + Tailwind CSS + Recharts</p>
            <p>Backend: FastAPI (async) with Redis-compatible TTL cache</p>
            <p>ML: PyTorch CMCL, PatchTST-style Transformer, FiLM Fusion</p>
            <p>Features: Welch PSD HRV, CGM variability, Sleep-stage alignment</p>
            <p>Calibration: Temperature scaling + ECE/Brier</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>AI/ML Details</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-muted">
            <p>Targets: ROC-AUC ≥ 0.94, F1 ≥ 0.91, p95 latency &lt; 120ms</p>
            <p>Horizon: 12h lookback → 12h forecast</p>
            <p>Fusion: z_fused = LayerNorm(MLP([h, z_s⊙z_t, |z_s-z_t|]) + h + attn)</p>
            <p>Loss: L_event + 0.8*L_risk + 0.4*L_ttx + 0.35*L_contrast</p>
            <p>CRI = 100*(0.55*p_event + 0.25*p_class + 0.12*strain + 0.08*σ(PRS))</p>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Architecture Flow</CardTitle></CardHeader>
          <CardContent className="text-sm text-muted leading-relaxed">
            <p>1. Data Ingestion → ECG/PPG/CGM/SpO₂/Skin Temp/Sleep (asynchronous)</p>
            <p>2. Processing Pipeline → 5-min grid (12 channels) + Welch PSD + personal baselines</p>
            <p>3. AI Inference Core → Dual-encoders + FiLM + Contrastive + Heads + Calibration</p>
            <p>4. Doctor Alert Dashboard → Triage, Evidence Chain, Actions, Explainability</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
