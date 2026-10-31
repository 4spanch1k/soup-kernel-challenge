"""Sequential same-session screen of optional optimizations; unchanged harness per run."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
FLAGS = ['SOUP_X_ATTN_CKPT', 'SOUP_X_DENSE_CKPT', 'SOUP_X_EXACT_RESID',
         'SOUP_X_MERGE_VARIANTS', 'SOUP_X_LORA', 'SOUP_X_FPRINT', 'SOUP_RECOMPUTE_MLP']
FAST = dict(SOUP_X_MERGE_VARIANTS='1', SOUP_X_LORA='1', SOUP_X_FPRINT='1')
MEM = dict(SOUP_X_ATTN_CKPT='1', SOUP_X_DENSE_CKPT='1', SOUP_X_EXACT_RESID='1')
CASES = [('control_1', {}), ('fast', FAST), ('attention', MEM),
         ('combined', dict(FAST, **MEM)), ('mlp', dict(FAST, SOUP_RECOMPUTE_MLP='1')),
         ('control_2', {})]
rows = []
for name, flags in CASES:
    env = os.environ.copy()
    for key in FLAGS:
        env.pop(key, None)
    env.update(flags)
    out = ROOT / 'runs_sweep' / name
    print(f'=== SCREEN {name} {flags} ===', flush=True)
    code = subprocess.call([sys.executable, '-u', 'harness/run_task.py', '--only', 'solution',
                            '--out', str(out)], env=env)
    result_path = out / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        summary = result.get('summary', {}).get('solution', {})
        rows.append(dict(name=name, flags=flags, status=code, **summary))
        print('SCREEN_RESULT ' + json.dumps(rows[-1]), flush=True)
        (ROOT / 'runs_sweep/summary.json').write_text(json.dumps(rows, indent=2))
    else:
        print(f'SCREEN_FAILED {name} status={code}', flush=True)
print('=== SCREEN COMPLETE ===', flush=True)
