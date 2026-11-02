"""Check local Markdown links and execute Python examples against the installed API."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote
import wakemarch

REPOSITORY_PREFIXES = ('https://github.com/haibuihoang/wakemarch/blob/main/',
                       'https://raw.githubusercontent.com/haibuihoang/wakemarch/main/')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    root = Path(__file__).resolve().parents[1]
    records = {}
    for path in sorted(root.rglob('*.md')):
        if any(part in ('build', '.venv') or part.endswith('.egg-info') for part in path.parts):
            continue
        source = path.read_text()
        for target in re.findall(r'\]\(([^)]+)\)', source):
            # Absolute links into this repository (used by README for PyPI) are checked locally.
            for prefix in REPOSITORY_PREFIXES:
                if target.startswith(prefix):
                    local = root/unquote(target[len(prefix):].split('#')[0])
                    if not local.exists():
                        raise AssertionError(f'{path}: broken repository link {target}')
                    break
            if '://' in target or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            if not (path.parent/target).exists():
                raise AssertionError(f'{path}: broken link {target}')
        namespace = {'__name__': '__documentation__'}
        blocks = re.findall(r'^```python\n(.*?)^```', source, flags=re.M | re.S)
        for index, block in enumerate(blocks):
            exec(compile(block, f'{path}:block-{index+1}', 'exec'), namespace)
        records[str(path.relative_to(root))] = dict(python_blocks=len(blocks), links_checked=True)
    args.output.write_text(json.dumps(dict(passed=True, version=wakemarch.__version__,
                                         origin=wakemarch.__file__, documents=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
