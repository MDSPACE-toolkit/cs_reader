"""Cross-check MDSPACE's cryoSPARC converter against UCSF pyem.

Run from the project folder:
    python -m pytest -v -s cs_reader_pytest.py

Requires cryosparc_P11_J4_003_particles.cs next to this test file.

Dependencies: pytest, numpy, scipy, cryosparc-tools, starfile, pyem
The pyem executable `csparc2star.py` must be on PATH.

IMPORTANT: This is a *cross-tool pose consistency* check, not a projection
or handedness test. Set CS_TEST_PASSTHROUGH to a companion .cs file when
pyem needs one; no particle stacks are required.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_converted_poses_against_pyem(tmp_path):
    cs = Path(__file__).resolve().parent / "cryosparc_P11_J4_003_particles.cs"
    assert cs.is_file(), f"Missing test dataset: {cs}"

    exe = shutil.which('csparc2star.py')
    if exe is None:
        pytest.fail('Install UCSF pyem and ensure csparc2star.py is on PATH')

    import cs_reader
    try:
        import starfile
    except ImportError:
        pytest.fail('Install starfile (pip install starfile)')

    mdspace_tsv = tmp_path / 'mdspace.tsv'
    star_path = tmp_path / 'pyem.star'
    cs_reader.convert(cs, mdspace_tsv)  # actual application entry point

    args = [exe, str(cs)]
    passthrough = os.environ.get('CS_TEST_PASSTHROUGH')
    if passthrough:
        args.append(str(Path(passthrough).expanduser().resolve()))
    args.append(str(star_path))
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, (
        f'pyem conversion failed (code {result.returncode}):\n'
        f'{result.stdout}\n{result.stderr}'
    )

    with mdspace_tsv.open(newline='') as handle:
        md_rows = list(csv.DictReader(handle, delimiter='\t'))
    star = starfile.read(star_path)
    if isinstance(star, dict):
        tables = [df for df in star.values()
                  if {'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'} <= set(df.columns)]
        assert len(tables) == 1, 'Expected one pyem particle table with Euler angles'
        star = tables[0]

    assert len(md_rows) == len(star), (
        f'Particle-count mismatch: MDSPACE={len(md_rows)}, pyem={len(star)}. '
        'Supply a matching passthrough .cs if needed.'
    )
    assert len(md_rows) > 0, 'Empty converted dataset'

    if 'rlnImageName' in star.columns:
        for i, (row, name) in enumerate(zip(md_rows, star['rlnImageName'])):
            prefix = str(name).split('@', 1)[0]
            if '@' in str(name) and row['image_index']:
                assert int(prefix) == int(row['image_index']), (
                    f'Particle order/index differs at row {i}: '
                    f'{row["image_index"]} vs {name}'
                )

    a = np.asarray([[float(r[f'xmd_angle_{key}']) for key in ('rot', 'tilt', 'psi')]
                    for r in md_rows], dtype=float)
    b = star[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy(dtype=float)

    md = Rotation.from_euler('ZYZ', a, degrees=True)
    candidates = {
        'RELION ZYZ as written': Rotation.from_euler('ZYZ', b, degrees=True),
        'RELION reversed order': Rotation.from_euler('ZYZ', b[:, ::-1], degrees=True),
        'RELION negated angles': Rotation.from_euler('ZYZ', -b, degrees=True),
        'RELION reversed and negated': Rotation.from_euler('ZYZ', -b[:, ::-1], degrees=True),
    }
    scores = []
    for name, r in candidates.items():
        for inverted, ref in ((False, r), (True, r.inv())):
            err = np.rad2deg((md.inv() * ref).magnitude())
            scores.append((float(np.max(err)), float(np.median(err)),
                           name + (' (inverse)' if inverted else ''), err))
    worst, median, convention, errors = min(scores, key=lambda x: x[0])
    print(f'\nCompared {len(a)} poses against pyem; best convention: {convention}')
    print(f'Median angular difference: {median:.8g} deg; max: {worst:.8g} deg')
    assert worst < 1e-3, (
        'MDSPACE poses disagree with pyem under the tested whole-dataset '
        f'conventions: max={worst:.6g} degrees, median={median:.6g}, '
        f'best={convention}. Check pose conventions, flags, and order. '
        'The candidate conventions may need a documented RELION-to-Xmipp '
        'mapping rather than treating this as proof of a converter bug.'
    )

