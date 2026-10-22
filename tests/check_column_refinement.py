"""Run the independent forward refinement check for the saved column JSON."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eitcore.physical import normalize_config
from eitcore.volume_benchmark import refinement_check

if __name__=='__main__':
    cfg=normalize_config(json.loads(Path('examples/column_104mm_256e.json').read_text()))
    result=refinement_check(cfg,Path(cfg['reconstruction']['output_directory']),log=lambda s:print(s,flush=True))
    print(json.dumps(result,indent=2),flush=True)
