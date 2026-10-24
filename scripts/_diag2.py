import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/"src")); sys.path.insert(0, str(Path.cwd()))
from biotwin.dataio import CohortBundle, build_tensors
from biotwin.features import WINDOW_CONTEXT_FEATURES
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

b = CohortBundle.load("demo"); t = build_tensors(b)
ids = b.patient_ids(); pi = {p:i for i,p in enumerate(ids)}
rng = np.random.default_rng(0); order = rng.permutation(len(ids))
val = [ids[i] for i in order[:10]]; tr = [ids[i] for i in order[10:]]
tri = np.where(np.isin(t.patient_index, [pi[p] for p in tr]))[0]
vai = np.where(np.isin(t.patient_index, [pi[p] for p in val]))[0]

k = len(WINDOW_CONTEXT_FEATURES)
ctx_only = t.X_static[:, -k:]
base_only = t.X_static[:, :-k]
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

for label, X in (("ctx20", ctx_only), ("clin+genomic", base_only), ("full", t.X_static)):
    print(f"--- {label} ---")
    Xp = prep.fit(X[tri]); Xt = Xp.transform(X[tri]); Xv = Xp.transform(X[vai])
    for cname, mk in (("logreg", lambda: LogisticRegression(max_iter=2000, C=0.1)),
                      ("mlp64", lambda: MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=400, random_state=0)),
                      ("gbm", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0))):
        line = f"   {cname:8s}"
        for k2, en in enumerate(("ischemia","crash","arrhythmia")):
            y = t.y_event[:, k2]
            m = mk().fit(Xt, y[tri])
            p = m.predict_proba(Xv)[:,1]
            line += f"  {en} {roc_auc_score(y[vai], p):.4f}"
        print(line)
print("positives per class:", t.y_event.sum(axis=0), "patients:", len(ids))
