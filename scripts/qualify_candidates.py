"""Full unchanged-harness qualification of shortlisted candidates on a second seed."""
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
CASES=[('int8_varlen',dict(SOUP_X_ACT_INT8='1',SOUP_X_VARLEN='1')),
       ('int8_varlen_norm',dict(SOUP_X_ACT_INT8='1',SOUP_X_VARLEN='1',SOUP_X_NORM_SAVE16='1'))]
results=[]
for name,flags in CASES:
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('SOUP_X_') or key=='SOUP_RECOMPUTE_MLP':env.pop(key)
 env.update(flags)
 out=ROOT/'runs_qualification'/name
 print('QUALIFY_START',name,flags,flush=True)
 status=subprocess.call([sys.executable,'-u','harness/run_task.py','--repeats','2','--seed','2026','--out',str(out)],env=env)
 check=subprocess.call([sys.executable,'-u','harness/check_rules.py','--result',str(out/'result.json')],env=env)
 row=dict(name=name,flags=flags,run_status=status,check_status=check)
 if (out/'result.json').exists():
  row['summary']=json.loads((out/'result.json').read_text()).get('summary')
  shutil.copyfile(out/'result.json',Path('/kaggle/working')/(name+'-result.json'))
 if (out/'checks.json').exists():
  row['checks']=json.loads((out/'checks.json').read_text())
  shutil.copyfile(out/'checks.json',Path('/kaggle/working')/(name+'-checks.json'))
 results.append(row)
 print('QUALIFY_RESULT',json.dumps(row),flush=True)
 Path('/kaggle/working/qualification-summary.json').write_text(json.dumps(results,indent=2))
with zipfile.ZipFile('/kaggle/working/qualification-evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in (ROOT/'runs_qualification').rglob('*'):
  if path.is_file() and path.suffix in ('.json','.jsonl','.yaml','.log') and not any('cache' in p for p in path.parts):
   archive.write(path,str(path.relative_to(ROOT)))
 for path in (ROOT/'solution').glob('*.py'):archive.write(path,str(path.relative_to(ROOT)))
print('QUALIFICATION_COMPLETE',flush=True)
