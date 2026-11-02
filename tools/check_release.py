"""Build and test wheel/sdist in fresh venvs; works outside the research checkout.

Default mode resolves dependencies from the package index in isolated venvs.
--offline inherits local libraries and is explicitly not a clean dependency test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--notebooks', action='store_true')
    p.add_argument('--minimum', action='store_true', help='Pin declared core floors for this verification')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_DEFAULT_TIMEOUT='15', PIP_RETRIES='0', PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env['MPLCONFIGDIR'] = str(out/'mplconfig')
    result = dict(passed=False, command=sys.argv, isolated_dependencies=not args.offline,
                  notebook_checks_requested=args.notebooks, minimum_dependencies=args.minimum, checks={}, artifacts={})
    def save():
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    def run(label, command, cwd):
        print(label, flush=True)
        with (out/f'{label}.log').open('w') as log:
            done = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        result['checks'][label] = done.returncode == 0
        save()
        if done.returncode:
            raise RuntimeError(f'{label} failed; see {out / (label+".log")}')
    save()
    with tempfile.TemporaryDirectory(prefix='wakemarch-release-') as temp:
        work = Path(temp)
        source = work/'source'
        shutil.copytree(root, source, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', '.ipynb_checkpoints', '.DS_Store', '*.egg-info', 'build', 'dist'))
        # Clean distribution copies, never the user's working notebooks.
        for notebook in (source/'examples').glob('*.ipynb'):
            content = json.loads(notebook.read_text())
            content.get('metadata', {}).pop('widgets', None)
            for cell in content['cells']:
                if cell['cell_type'] == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
                    cell.get('metadata', {}).pop('execution', None)
            notebook.write_text(json.dumps(content, indent=1)+'\n')
        dist = out/'dist'
        dist.mkdir()
        run('build', [sys.executable, '-I', '-B', '-c',
            'import sys; from setuptools.build_meta import build_sdist, build_wheel; '
            'destination=sys.argv[1]; build_sdist(destination); build_wheel(destination)', dist], source)
        wheel, sdist = next(dist.glob('*.whl')), next(dist.glob('*.tar.gz'))
        for artifact in (wheel, sdist):
            result['artifacts'][artifact.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as archive:
            assert all(name.startswith('wakemarch/') or '.dist-info/' in name for name in archive.namelist())
        with tarfile.open(sdist) as archive:
            assert not any('.ipynb_checkpoints' in n or '__pycache__' in n or n.endswith('.DS_Store') for n in archive.getnames())
            assert sum(n.endswith('.ipynb') for n in archive.getnames()) == 4
            archive.extractall(work/'extracted', filter='data')
        extracted = next((work/'extracted').iterdir())
        required = ['README.md', 'CITATION.md', 'CITATION.cff', 'CHANGELOG.md', 'LICENSE', 'docs/USER_GUIDE.md', 'docs/SCIENCE.md',
                    'docs/RELEASING.md', 'docs/release-metadata.json', 'docs/images/wake.png',
                    'tools/check_release.py', 'tools/check_docs.py', 'tools/run_notebooks.py',
                    'examples/wakemarch_api_test.ipynb', 'examples/wakemarch_performance.ipynb',
                    'examples/wakemarch_piv_validation.ipynb', 'examples/wakemarch_turbines.ipynb',
                    'src/wakemarch/data/nrel_5MW.json', 'src/wakemarch/data/iea_10MW.json',
                    'src/wakemarch/data/iea_15MW.json', 'src/wakemarch/data/FLORIS_LICENSE.txt']
        assert all((extracted/name).is_file() for name in required)
        for notebook in (extracted/'examples').glob('*.ipynb'):
            assert all(not c.get('outputs') and c.get('execution_count') is None
                       for c in json.loads(notebook.read_text())['cells'] if c['cell_type'] == 'code')
        result['checks']['artifact_contents'] = True
        for kind, artifact in [('wheel', wheel), ('sdist', sdist)]:
            prefix = work/f'{kind}_venv'
            venv.EnvBuilder(with_pip=True, system_site_packages=args.offline).create(prefix)
            python = prefix/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            command = [python, '-I', '-m', 'pip', 'install']
            if args.offline:
                command += ['--no-index', '--no-deps', '--no-build-isolation']
            if args.minimum:
                constraints = out/'minimum.txt'
                constraints.write_text('numpy==1.24.4\nscipy==1.10.1\n')
                command += ['--constraint', constraints]
            extra = '[notebooks]' if args.notebooks and kind == 'wheel' else ''
            run(f'{kind}_install', command+[str(artifact)+extra], work)
            # -I and a cwd outside the source prevent accidental checkout imports.
            run(f'{kind}_origins', [python, '-I', '-c',
                'import pathlib,sys,importlib.metadata as m,numpy,scipy,wakemarch; '
                'prefix=pathlib.Path(sys.prefix).resolve(); '
                'modules=[wakemarch]'+('' if args.offline else '+[numpy,scipy]')+'; '
                'assert all(pathlib.Path(x.__file__).resolve().is_relative_to(prefix) for x in modules); '
                'print(sys.version); print([(x.__name__,x.__version__,x.__file__) for x in (wakemarch,numpy,scipy)]); '
                'print(sorted((d.metadata["Name"],d.version) for d in m.distributions()))'], work)
            if not args.offline:
                run(f'{kind}_pip_check', [python, '-I', '-m', 'pip', 'check'], work)
            run(f'{kind}_api', [python, '-I', '-B', extracted/'tests/test_api.py'], work)
            run(f'{kind}_turbine', [python, '-I', '-B', extracted/'tests/test_turbine.py'], work)
            run(f'{kind}_example', [python, '-I', '-B', extracted/'examples/first_wake.py'], work)
            run(f'{kind}_docs', [python, '-I', '-B', extracted/'tools/check_docs.py', '--output', out/f'{kind}_docs.json'], work)
            if args.notebooks and kind == 'wheel':
                run('figure', [python, '-I', '-B', extracted/'examples/plot_wake.py', '--output', out/'figure'], work)
                run('kernel', [python, '-I', '-m', 'ipykernel', 'install', '--prefix', prefix, '--name', 'wakemarch-check'], work)
                run('notebooks', [python, '-I', '-B', extracted/'tools/run_notebooks.py', '--kernel', 'wakemarch-check', '--output', out/'notebooks'], work)
        result['passed'] = all(result['checks'].values())
        save()


if __name__ == '__main__':
    main()
