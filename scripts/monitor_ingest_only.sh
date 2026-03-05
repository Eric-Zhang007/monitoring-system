#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/c/Users/zjc01/monitoring-system"
cd "$ROOT_DIR"

while true; do
  python3 - <<'PY'
import json, os, re, math, subprocess
from datetime import datetime, timezone

root = '/mnt/c/Users/zjc01/monitoring-system'
frames = ['15m','30m','1h','4h','6h','12h','1d','5m','1m']
ckpt_tpl = os.path.join(root, 'artifacts/checkpoints/top11_50_crypto', 'ingest_perp_{tf}_top11_50_full.json')

now = datetime.now(timezone.utc)
print('=' * 100, flush=True)
print(f"monitor_at={now.isoformat().replace('+00:00','Z')}", flush=True)

ps = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True, check=True)
active = []
for line in ps.stdout.splitlines()[1:]:
    if 'scripts/ingest_bitget_market_bars.py' not in line:
        continue
    m = re.match(r'\s*(\d+)\s+(.*)$', line)
    if not m:
        continue
    pid = int(m.group(1)); args = m.group(2)
    tfm = re.search(r'--timeframe\s+([^\s]+)', args)
    ckm = re.search(r'--checkpoint-file\s+([^\s]+)', args)
    timeframe = (tfm.group(1) if tfm else 'unknown')
    if timeframe not in {'1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d'}:
        continue
    active.append({
        'pid': pid,
        'timeframe': timeframe,
        'checkpoint': (ckm.group(1) if ckm else 'n/a'),
    })

print('1) active_ingest_processes', flush=True)
if active:
    for p in sorted(active, key=lambda x: (x['timeframe'], x['pid'])):
        print(f"- pid={p['pid']} timeframe={p['timeframe']} checkpoint={p['checkpoint']}", flush=True)
else:
    print('- none', flush=True)

print('2) top11_50_checkpoint_progress', flush=True)
progress = []
unfinished = []
for tf in frames:
    path = ckpt_tpl.format(tf=tf)
    rec = {
        'timeframe': tf,
        'completed_chunks': 0,
        'updated_at': 'missing',
        'failed_chunks': 0,
        'unfinished': True,
    }
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                obj = json.load(fh)
            cc = obj.get('completed_chunks') or {}
            rec['completed_chunks'] = len(cc) if isinstance(cc, dict) else 0
            rec['updated_at'] = obj.get('updated_at') or 'missing'
            fc = obj.get('failed_chunks')
            rec['failed_chunks'] = len(fc) if isinstance(fc, (list, dict)) else 0
            rs = obj.get('run_spec') or {}
            s = rs.get('start_ms'); e = rs.get('end_ms'); cd = rs.get('chunk_days')
            if isinstance(s, int) and isinstance(e, int) and isinstance(cd, (int, float)) and cd > 0 and e > s:
                exp = math.ceil((e - s) / int(cd * 86400000))
                rec['unfinished'] = rec['completed_chunks'] < exp
            else:
                rec['unfinished'] = rec['completed_chunks'] == 0
        except Exception:
            rec['unfinished'] = True
    progress.append(rec)
    if rec['unfinished']:
        unfinished.append(tf)
    print(f"- {tf}: completed_chunks={rec['completed_chunks']} updated_at={rec['updated_at']} failed_chunks={rec['failed_chunks']}", flush=True)

print('3) stall_detection', flush=True)
active_tfs = {a['timeframe'] for a in active}
stalled = []
for rec in progress:
    if rec['timeframe'] not in active_tfs:
        continue
    raw = rec.get('updated_at')
    if not raw or raw == 'missing':
        stalled.append((rec['timeframe'], 'missing_updated_at'))
        continue
    try:
        dt = datetime.fromisoformat(str(raw).replace('Z', '+00:00')).astimezone(timezone.utc)
        age = (now - dt).total_seconds()
        if age > 300:
            stalled.append((rec['timeframe'], f'{int(age)}s_old'))
    except Exception:
        stalled.append((rec['timeframe'], 'invalid_updated_at'))
if stalled:
    for tf, reason in stalled:
        print(f"- STALL timeframe={tf} reason={reason}", flush=True)
else:
    print('- no_stall', flush=True)

print('4) process_vs_checkpoint_alert', flush=True)
if (not active) and unfinished:
    print(f"- ALERT no_ingest_process_and_unfinished_checkpoints={','.join(unfinished)}", flush=True)
else:
    print('- ok', flush=True)
PY
  sleep 120
done
