"""Screen independently written CE and MLP saved-activation candidates."""
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
FLAGS=['SOUP_X_PARALLEL_CE','SOUP_X_ACT_INT8','SOUP_X_HEAD_CHUNK','SOUP_X_ATTN_CKPT',
       'SOUP_X_DENSE_CKPT','SOUP_X_EXACT_RESID','SOUP_X_MERGE_VARIANTS','SOUP_X_LORA',
       'SOUP_X_FPRINT','SOUP_RECOMPUTE_MLP']
FAST=dict(SOUP_X_MERGE_VARIANTS='1',SOUP_X_LORA='1',SOUP_X_FPRINT='1')
CASES=[('control1',{}),('parallel',dict(FAST,SOUP_X_PARALLEL_CE='1')),
       ('parallel512',dict(FAST,SOUP_X_PARALLEL_CE='1',SOUP_X_HEAD_CHUNK='512')),
       ('int8',dict(FAST,SOUP_X_PARALLEL_CE='1',SOUP_X_ACT_INT8='1')),
       ('int8_attention',dict(FAST,SOUP_X_PARALLEL_CE='1',SOUP_X_ACT_INT8='1',
                              SOUP_X_ATTN_CKPT='1',SOUP_X_DENSE_CKPT='1',SOUP_X_EXACT_RESID='1')),
       ('control2',{})]
rows=[]
for name,flags in CASES:
 env=os.environ.copy()
 for key in FLAGS: env.pop(key,None)
 env.update(flags)
 out=ROOT/'runs_stage2'/name
 print('STAGE2_START',name,flags,flush=True)
 status=subprocess.call([sys.executable,'-u','harness/run_task.py','--only','solution','--out',str(out)],env=env)
 if (out/'result.json').exists():
  result=json.loads((out/'result.json').read_text())
  row=dict(name=name,flags=flags,status=status,**result.get('summary',{}).get('solution',{}))
  rows.append(row)
  print('STAGE2_RESULT',json.dumps(row),flush=True)
  (ROOT/'runs_stage2/summary.json').write_text(json.dumps(rows,indent=2))
print('STAGE2_COMPLETE',flush=True)
