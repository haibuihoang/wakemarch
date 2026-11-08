# Development and release verification

## Portable checks

From the package root, in an environment with Python 3.11+ and setuptools/wheel:

```sh
python -m pip install 'setuptools>=77' wheel
python -I -B tools/check_release.py --output /tmp/wakemarch-check --notebooks
```

Use any new writable output directory on Windows instead of `/tmp/...`. The
driver builds a wheel and sdist, creates fresh isolated environments, resolves
dependencies, checks import origins, runs `pip check`, tests, the example and
executable documentation. It checks the built source contains the guide and
notebooks. `--notebooks` adds figure generation and fresh-kernel execution from
the extracted source using the installed wheel. Outputs and logs stay outside
the source tree. `--minimum` pins the declared floors, NumPy 1.24.4 and
SciPy 1.10.1.

For an offline local check only:

```sh
python -I -B tools/check_release.py --offline --notebooks --output /tmp/wakemarch-offline
```

This mode inherits local numerical/notebook/build dependencies; it is **not**
fresh dependency resolution. Its result records `isolated_dependencies=false`.
The authors additionally compare every release with the research solver on
eight cases (`scripts/check_package_030rc2.py` of the paper companion); the
research solver is not shipped.

## Continuous integration

`.github/workflows/package.yml` runs wheel/sdist checks on Linux, macOS and
Windows with Python 3.11–3.14, plus a Linux Python 3.11 minimum-dependency job.
The Linux Python 3.11 job also executes the data-free notebooks. Dependency
upper bounds are not evidence of compatibility with every future version.
The workflow has read-only repository access.

## Publishing

`.github/workflows/publish.yml` builds the distributions and uploads them with
PyPI trusted publishing (no API tokens): manually to TestPyPI
(`workflow_dispatch`, target `testpypi`) or to PyPI when a GitHub release is
published. The trusted publisher must be configured on (Test)PyPI for this
repository, workflow `publish.yml` and environments `testpypi`/`pypi`.
The workflow refuses to run while any release file still contains the marker
`PENDING-D2` (author/copyright metadata not yet supplied) or the classifier
`Private :: Do Not Upload`.

Release steps: update `version`, `CHANGELOG.md` and `CITATION.cff`; run the
portable checks (normal, `--minimum`, `--notebooks`); confirm CI is green;
rehearse on TestPyPI; then publish a GitHub release.

This follows the [GitHub Python CI guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python),
the [PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [PyPA trusted publishing](https://docs.pypi.org/trusted-publishers/).

## Paper reproducibility

The paper's figures were produced with the authors' research solver. Record the
package version used for any figure only after reproducing it with that version.
`docs/release-metadata.json` records licence, repository and DOI metadata.

## Notebooks

To run all four notebooks with a registered local kernel and an existing
download of the PIV data (no automatic data access):

```sh
python -I -B tools/run_notebooks.py --output /tmp/wakemarch-notebooks --piv-data /path/to/Bastankhah_jfm_2016.h5
```

The CI notebook job executes only the three data-free examples. The PIV notebook
is checked locally against the checksum-pinned public file; this does not create
a redistributable data fixture or a blind-validation claim.
