"""Execute clean example notebooks using a registered kernel; save outputs separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kernel', default='python3')
    p.add_argument('--piv-data', type=Path, help='Optional pinned Zenodo 3607598 file; also execute measurement notebook')
    args = p.parse_args()
    if args.piv_data:
        if not args.piv_data.is_file():
            p.error('PIV file does not exist')
        os.environ['WAKEMARCH_PIV_DATA'] = str(args.piv_data.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key in ('IPYTHONDIR', 'JUPYTER_CONFIG_DIR', 'JUPYTER_RUNTIME_DIR', 'MPLCONFIGDIR'):
        path = out/key.lower()
        path.mkdir()
        os.environ[key] = str(path)
    os.environ['JUPYTER_PATH'] = os.pathsep.join(str(Path(p)/'share/jupyter') for p in (sys.prefix, sys.base_prefix))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    records = {}
    for source in sorted((Path(__file__).resolve().parents[1]/'examples').glob('*.ipynb')):
        if source.stem == 'wakemarch_piv_validation' and not args.piv_data:
            continue
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        nb = nbformat.read(source, as_version=4)
        # Preserve the user's working notebook; discard saved output only in
        # the in-memory copy so every check starts from fresh execution.
        for cell in nb.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        try:
            NotebookClient(nb, timeout=300, kernel_name=args.kernel,
                           resources={'metadata': {'path': str(out)}}).execute()
        finally:
            nbformat.write(nb, out/f'{source.stem}.executed.ipynb')
        html, _ = HTMLExporter().from_notebook_node(nb)
        (out/f'{source.stem}.html').write_text(html)
        assert before == hashlib.sha256(source.read_bytes()).hexdigest()
        assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code' and c.source.strip())
        outputs = [o for c in nb.cells for o in c.get('outputs', [])]
        assert not any(o.output_type == 'error' for o in outputs)
        bars = [o['data']['text/html'] for o in outputs if '<progress ' in o.get('data', {}).get('text/html', '')]
        expected_bars, expected_figures = {'wakemarch_api_test': (4, 2), 'wakemarch_performance': (3, 1), 'wakemarch_turbines': (1, 2), 'wakemarch_piv_validation': (3, 1)}[source.stem]
        assert len(bars) == expected_bars and all('Complete' in b and 'value="1.000000"' in b for b in bars)
        assert sum('image/png' in o.get('data', {}) for o in outputs) == expected_figures
        records[source.name] = dict(passed=True, source_sha256=before,
            code_cells=sum(c.cell_type == 'code' for c in nb.cells), completed_bars=len(bars), figures=expected_figures)
        (out/'results.json').write_text(json.dumps(dict(passed=False, notebooks=records), indent=2)+'\n')
    assert len(records) == (4 if args.piv_data else 3)
    (out/'results.json').write_text(json.dumps(dict(passed=True, notebooks=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
