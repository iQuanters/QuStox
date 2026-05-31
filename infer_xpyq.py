#!/usr/bin/env python3
"""
infer_xpyq.py — HQRC inference with the reservoir run on the xpyq cloud platform.
================================================================================
Same idea as infer.py (pick rows by date/time, evolve each price window through
the fixed 16-qubit Ising reservoir, classify with the SAVED readout), but the
reservoir circuits execute on the xpyq Playground Server
(https://xpyq-lib-production.up.railway.app) instead of a local simulator:

    Close_prev window ─▶ build qiskit circuit ─▶ emit as `code` string
                       ─▶ xpyq /quantum/call ─▶ counts ─▶ 40 features
    40 feats + Open/High/Low ─▶ 43 ─▶ saved MLP ─▶ buy/sell + confidence

Two execution modes (--mode):
  * sim  (default) : tool=`simulate` (aer) on xpyq — FREE, validates the whole
                     cloud round-trip and the readout end-to-end.
  * qpu            : tool=`qpu_submit` on a REAL device. BILLABLE. Two-step flow:
                     by default this only PRINTS a cost estimate and stops; it
                     submits for real ONLY when you pass --confirm.

Auth: set XPYQ_API_KEY (preferred) or pass --api-key. Reuses the circuit builder
and readout/dataset/row-selection helpers from infer.py (no retraining).

Examples
--------
    export XPYQ_API_KEY=xpyq_live_xxx
    python infer_xpyq.py 2026-05-29                 # whole day, xpyq sim (free)
    python infer_xpyq.py --index -1                 # last bar, xpyq sim
    python infer_xpyq.py --index -1 --mode qpu --backend ibm_marrakesh
                                                    # ^ prints cost estimate only
    python infer_xpyq.py --index -1 --mode qpu --backend ibm_marrakesh --confirm
                                                    # ^ actually spends credits
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from infer import (
    QuantumReservoir, load_readout, load_dataset, select_rows, class_confidence,
)

XPYQ_BASE = os.environ.get("XPYQ_BASE_URL", "https://xpyq-lib-production.up.railway.app")
CALL_EP   = "/api/v2/quantum/call"


# ============================================================ circuit → qiskit code
def circuit_to_code(qc) -> str:
    """Serialize a QuantumCircuit to standalone qiskit source defining `qc`
    (angles baked in, so the worker runs exactly this circuit — no RNG / numpy
    version dependence). measure_all() is re-added at the end."""
    lines = ["from qiskit import QuantumCircuit", f"qc = QuantumCircuit({qc.num_qubits})"]
    for instr in qc.data:
        name = instr.operation.name
        if name in ("measure", "barrier"):
            continue                                   # measure_all re-added below
        qidx = [qc.find_bit(q).index for q in instr.qubits]
        params = ", ".join(repr(float(p)) for p in instr.operation.params)
        qargs = ", ".join(str(q) for q in qidx)
        lines.append(f"qc.{name}({params + ', ' if params else ''}{qargs})")
    lines.append("qc.measure_all()")
    return "\n".join(lines)


# ============================================================ xpyq client
class XpyqClient:
    def __init__(self, api_key: str, ibm_token: str = "", base: str = XPYQ_BASE, timeout: int = 120):
        if not api_key:
            raise SystemExit("[xpyq] no API key — set XPYQ_API_KEY or pass --api-key")
        self.base, self.timeout = base.rstrip("/"), timeout
        self.h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # IBM real-QPU is BYOK: the IBM Quantum token is sent per-request, never persisted.
        self.ibm_h = {**self.h, "X-IBM-Token": ibm_token} if ibm_token else None

    def _call(self, tool: str, arguments: dict) -> dict:
        r = requests.post(f"{self.base}{CALL_EP}", headers=self.h,
                          json={"tool": tool, "arguments": arguments}, timeout=self.timeout)
        r.raise_for_status()
        env = r.json()
        if env.get("isError"):
            raise SystemExit(f"[xpyq] {tool} error: {env.get('content')}")
        content = env.get("content", [])
        payload = content[0] if content else "{}"
        out = json.loads(payload) if isinstance(payload, str) else payload
        # NOTE: don't treat success=False as fatal here — the qpu_submit ESTIMATE
        # legitimately returns success=False + requires_confirmation. Callers decide.
        return out

    def credits(self) -> int | None:
        try:
            r = requests.get(f"{self.base}/api/v2/auth/credits", headers=self.h, timeout=30)
            return r.json().get("balance")
        except Exception:
            return None

    def simulate(self, code: str, shots: int, seed: int | None) -> dict:
        out = self._call("simulate", {"code": code, "method": "qasm",
                                      "shots": shots, "seed_simulator": seed, "backend": "aer"})
        if not out.get("success", True):
            raise SystemExit(f"[xpyq] simulate failed: {out}")
        return out["counts"]

    # ---- real IBM QPU (BYOK: X-IBM-Token sent per request) ----------------------
    def _ibm_post(self, path: str, body: dict) -> dict:
        if not self.ibm_h:
            raise SystemExit("[xpyq] real-QPU needs an IBM token — set IBM_QUANTUM_TOKEN or pass --ibm-token")
        r = requests.post(f"{self.base}{path}", headers=self.ibm_h, json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, dict) and out.get("isError"):
            raise SystemExit(f"[xpyq] {path} error: {out.get('content')}")
        return out

    def ibm_devices(self) -> list[dict]:
        if not self.ibm_h:
            raise SystemExit("[xpyq] real-QPU needs an IBM token — set IBM_QUANTUM_TOKEN or pass --ibm-token")
        r = requests.get(f"{self.base}/api/v2/quantum/ibm/devices", headers=self.ibm_h, timeout=30)
        r.raise_for_status()
        return r.json().get("backends", [])

    def ibm_submit(self, code: str, backend: str | None, shots: int) -> dict:
        """Submit to a real IBM QPU via Qiskit Runtime → {job_id, backend, shots}.
        Billed against your IBM Open-Plan quota, not xpyq credits."""
        body = {"code": code, "source_framework": "qiskit", "shots": shots}
        if backend:
            body["backend"] = backend
        out = self._ibm_post("/api/v2/quantum/ibm/submit", body)
        if not out.get("job_id"):
            raise SystemExit(f"[xpyq] IBM submit returned no job_id: {out}")
        return out

    def ibm_status(self, job_id: str) -> dict:
        return self._ibm_post("/api/v2/quantum/ibm/status", {"job_id": job_id})

    def ibm_wait(self, job_id: str, poll: int = 15, timeout: int = 3600) -> dict:
        t0 = time.time()
        while True:
            st = self.ibm_status(job_id)
            state = str(st.get("status", "")).upper()
            print(f"[xpyq]   job {job_id}: {state or st}")
            if state in ("DONE", "COMPLETED", "SUCCESS"):
                return st
            if state in ("ERROR", "FAILED", "CANCELLED", "CANCELED"):
                raise SystemExit(f"[xpyq] IBM job {job_id} ended: {st}")
            if time.time() - t0 > timeout:
                raise SystemExit(f"[xpyq] IBM job {job_id} timed out after {timeout}s")
            time.sleep(poll)


def _be_to_le(counts: dict) -> dict:
    """xpyq returns bitstrings big-endian (qubit 0 leftmost); qiskit get_counts —
    which QuantumReservoir._counts_to_features assumes — is little-endian (qubit 0
    rightmost). Reverse each key so qubit indices line up with the local reservoir
    the readout was trained on. (Verified: features then match local Aer at corr≈1.)"""
    return {k.replace(" ", "")[::-1]: v for k, v in counts.items()}


def _counts_from_status(st: dict) -> dict:
    """Pull the counts dict out of an IBM status result (DONE → {'counts': {...}})."""
    v = st.get("counts")
    if isinstance(v, dict) and v:
        return v
    if isinstance(v, dict) and "counts" in v:           # nested fallback
        return v["counts"]
    raise SystemExit(f"[xpyq] could not find counts in job result: {st}")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="HQRC inference with the reservoir on the xpyq cloud platform.")
    ap.add_argument("when", nargs="*", help="one timestamp/date, or two for an inclusive range")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--all-day", action="store_true")
    ap.add_argument("--index", type=int, nargs="+", help="select by raw row index (negatives ok)")
    ap.add_argument("--dataset", default="AAPL_10m_v2.csv")
    ap.add_argument("--readout", default="readout_weights")
    ap.add_argument("--mode", choices=["sim", "qpu"], default="sim",
                    help="sim = xpyq simulator (free); qpu = real IBM QPU (BYOK, uses your IBM quota)")
    ap.add_argument("--backend", default="ibm_marrakesh",
                    help="IBM backend (--mode qpu); blank/'' = least-busy")
    ap.add_argument("--shots", type=int, help="shots (default from readout meta)")
    ap.add_argument("--confirm", action="store_true",
                    help="REQUIRED to actually submit to real hardware in --mode qpu")
    ap.add_argument("--api-key", default=os.environ.get("XPYQ_API_KEY", ""))
    ap.add_argument("--ibm-token", default=os.environ.get("IBM_QUANTUM_TOKEN", ""),
                    help="IBM Quantum token for real-QPU runs (or set IBM_QUANTUM_TOKEN)")
    args = ap.parse_args()

    # 1. saved readout + reservoir config it was trained with --------------------
    pipe, meta = load_readout(args.readout)
    window     = int(meta.get("reservoir_window_n", 12))
    encode_eff = float(meta.get("eff_encode_scale", 1.0))
    seed       = int(meta.get("seed", 42))
    shots      = args.shots or int(meta.get("shots", 1000))
    expect_in  = int(meta.get("n_input_features", 43))
    print(f"\n{'='*72}\n  HQRC inference via xpyq — mode={args.mode}"
          f"{(' backend='+args.backend) if args.mode=='qpu' else ''}\n{'='*72}")
    print(f"[readout] window={window}  encode_scale={encode_eff:.4f}  seed={seed}  shots={shots}")

    # 2. data + row selection ----------------------------------------------------
    df, prev_cols, ohl_cols, ts_wall = load_dataset(args.dataset)
    rows = select_rows(df, ts_wall, args)
    X   = df[prev_cols].to_numpy(dtype=np.float64)[rows]
    OHL = df[ohl_cols].to_numpy(dtype=np.float64)[rows] if ohl_cols else None
    print(f"[select] {len(rows)} row(s) from {args.dataset}")

    # 3. build the reservoir circuits (locally) ----------------------------------
    res = QuantumReservoir(window=window, shots=shots, encode_scale=encode_eff, seed=seed)
    codes = [circuit_to_code(res.build_circuit(X[i])) for i in range(len(rows))]

    client = XpyqClient(args.api_key, ibm_token=args.ibm_token)
    print(f"[xpyq] connected — credit balance: {client.credits()}")

    # 4. run the reservoir on xpyq → counts → 40 features ------------------------
    feat_list = []
    if args.mode == "sim":
        print(f"[xpyq] simulating {len(codes)} circuit(s) (aer, free) …")
        for i, code in enumerate(codes):
            counts = client.simulate(code, shots=shots, seed=seed)
            feat_list.append(res._counts_to_features(_be_to_le(counts)))
            print(f"[xpyq]   {i+1}/{len(codes)} done")
    else:
        # --- real IBM QPU (BYOK): preview the device, submit only with --confirm -
        backend = args.backend or None
        devs = {d["name"]: d for d in client.ibm_devices()}
        if backend and backend not in devs:
            raise SystemExit(f"[xpyq] backend {backend} not available; have: {list(devs)}")
        tgt = devs.get(backend)
        dev_info = f"({tgt['qubits']}q, {tgt['pending_jobs']} pending) " if tgt else ""
        print(f"\n[xpyq] real IBM QPU (BYOK → your IBM quota, not credits)")
        print(f"       backend={backend or 'least-busy'}  {dev_info}"
              f"circuits={len(codes)}  shots={shots}")
        if not args.confirm:
            print("[xpyq] DRY RUN — pass --confirm to actually submit to real hardware.")
            print("       (one window = one Runtime job, billed to your IBM Open-Plan quota)\n")
            return
        for i, code in enumerate(codes):
            job = client.ibm_submit(code, backend=backend, shots=shots)
            jid = job["job_id"]
            print(f"[xpyq] submitted circuit {i+1}/{len(codes)} → job {jid} on {job.get('backend')}")
            st = client.ibm_wait(jid)
            feat_list.append(res._counts_to_features(_be_to_le(_counts_from_status(st))))

    Q = np.stack(feat_list)
    feats = Q if OHL is None else np.hstack([Q, OHL])
    if feats.shape[1] != expect_in:
        print(f"[warn] built {feats.shape[1]} features but readout expects {expect_in}")

    # 5. classify with the saved readout -----------------------------------------
    mlp = pipe.named_steps["mlpclassifier"]
    buy_col = list(mlp.classes_).index(1)
    proba = pipe.predict_proba(feats)[:, buy_col]
    pred = (proba >= 0.5).astype(int)
    conf = class_confidence(proba)
    has_truth = "target" in df.columns
    if has_truth:
        y = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()[rows]

    # 6. report ------------------------------------------------------------------
    print("\n" + "=" * 80)
    hdr = f"  {'timestamp':<21}{'signal':>8}{'P(buy)':>9}{'conf%':>8}"
    if has_truth:
        hdr += f"{'actual':>9}{'hit':>5}"
    print(hdr); print("-" * 80)
    for k, r in enumerate(rows):
        when = str(ts_wall.iloc[r]) if ts_wall is not None else f"row {r}"
        sig = "BUY" if pred[k] == 1 else "SELL"
        line = f"  {when:<21}{sig:>8}{proba[k]:>9.3f}{conf[k]:>8.1f}"
        if has_truth:
            line += f"{('buy' if y[k] else 'sell'):>9}{('✓' if pred[k]==y[k] else '✗'):>5}"
        print(line)
    print("=" * 80)
    if has_truth and len(rows) > 1:
        print(f"  accuracy over {len(rows)} rows: {float((pred==y).mean()):.1%}")
    print(f"  xpyq balance after run: {client.credits()}\n")


if __name__ == "__main__":
    main()
