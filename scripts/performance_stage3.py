"""T4 screen of output ownership, startup overhead, and denser saved activations."""
import json, os, subprocess, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)

def clean_env():
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('SOUP_X_') or key=='SOUP_RECOMPUTE_MLP':env.pop(key)
 return env

test_status={}
for name,files in [('lora',['solution/test_lora_output_storage.py']),
                   ('codec',['solution/test_activation_codec.py']),
                   ('norm',['solution/test_norm_storage.py'])]:
 test_status[name]=subprocess.call([sys.executable,'-m','pytest',*files,'-q','-x','-p','no:cacheprovider'],env=clean_env())
print('STAGE3_TESTS',json.dumps(test_status),flush=True)
Path('/kaggle/working/stage3-tests.json').write_text(json.dumps(test_status,indent=2))
norm=dict(SOUP_X_NORM_SAVE16='1') if test_status['norm']==0 else {}
control=dict(norm,SOUP_X_ACT_INT8='1')
cases=[('control1',control)]
fast=dict(control)
if test_status['lora']==0:
 fast.update(SOUP_X_LINEAR_OUT='1',SOUP_X_LORA='1')
 cases.append(('lora',dict(fast)))
fast.update(SOUP_X_FPRINT='1',SOUP_X_MERGE_VARIANTS='1')
cases.append(('startup',dict(fast)))
if test_status['codec']==0:
 four=dict(fast)
 four.pop('SOUP_X_ACT_INT8',None)
 four['SOUP_X_ACT_INT4']='1'
 cases.append(('int4',four))
cases.append(('control2',control))
rows=[]
for name,flags in cases:
 env=clean_env();env.update(flags)
 out=ROOT/'runs_stage3'/name
 print('STAGE3_START',name,flags,flush=True)
 status=subprocess.call([sys.executable,'-u','harness/run_task.py','--only','solution','--out',str(out)],env=env)
 row=dict(name=name,flags=flags,status=status)
 if (out/'result.json').exists():
  result=json.loads((out/'result.json').read_text())
  row.update(result.get('summary',{}).get('solution',{}))
 rows.append(row)
 print('STAGE3_RESULT',json.dumps(row),flush=True)
 if not row.get('ok'):
  for log in out.glob('solution_*/train.log'):print(log.read_text()[-12000:],flush=True)
 Path('/kaggle/working/stage3-summary.json').write_text(json.dumps(rows,indent=2))
with zipfile.ZipFile('/kaggle/working/stage3-evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in (ROOT/'runs_stage3').rglob('*'):
  if path.is_file() and path.suffix in ('.json','.jsonl','.yaml','.log') and not any('cache' in p for p in path.parts):archive.write(path,str(path.relative_to(ROOT)))
print('STAGE3_COMPLETE',flush=True)
