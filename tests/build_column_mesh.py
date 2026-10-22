"""Build/inspect the configured column meshes without starting inversion."""
import json
import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mesh_tools import build_mesh, mesh_info
from eitcore.physical import normalize_config

if __name__ == '__main__':
    cfg = normalize_config(json.loads(pathlib.Path('examples/column_104mm_256e.json').read_text()))
    out = pathlib.Path(cfg['reconstruction']['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    for name in ('truth', 'reconstruction'):
        m = dict(cfg['mesh'])
        if name != 'truth':
            m.update({k: cfg['reconstruction'][k] for k in ('h_center','h_edge','h_band','electrode_mesh_size')})
        start = time.perf_counter()
        print(f'Building {name}: {m}', flush=True)
        path = build_mesh(m, verbose=False)
        info = mesh_info(path, dimension=3)
        info['build_seconds'] = time.perf_counter()-start
        (out/f'{name}_mesh_info.json').write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
