#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from cryosparc.dataset import Dataset


def scalar_value(value: Any) -> str | None:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0].item()
            else:
                return None

        if isinstance(value, np.generic):
            value = value.item()

    except Exception:
        pass

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        return re.sub(r"\s+", "_", text)

    return None


def vector_value(value: Any, size: int) -> list[float] | None:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)

            if flat.size < size:
                return None

            return [float(flat[index]) for index in range(size)]

        if isinstance(value, np.generic):
            if size == 1:
                return [float(value.item())]

            return None

    except Exception:
        pass

    try:
        if len(value) < size:
            return None

        return [float(value[index]) for index in range(size)]

    except Exception:
        return None


def rotation_vector_to_matrix(
    rotation_vector: list[float],
) -> list[list[float]]:
    x, y, z = rotation_vector

    theta = math.sqrt(
        x * x +
        y * y +
        z * z
    )

    if theta == 0.0:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    x /= theta
    y /= theta
    z /= theta

    cosine = math.cos(theta)
    sine = math.sin(theta)
    one_minus_cosine = 1.0 - cosine

    return [
        [
            one_minus_cosine * x * x + cosine,
            one_minus_cosine * x * y - sine * z,
            one_minus_cosine * x * z + sine * y,
        ],
        [
            one_minus_cosine * x * y + sine * z,
            one_minus_cosine * y * y + cosine,
            one_minus_cosine * y * z - sine * x,
        ],
        [
            one_minus_cosine * x * z - sine * y,
            one_minus_cosine * y * z + sine * x,
            one_minus_cosine * z * z + cosine,
        ],
    ]


def matrix_to_zyz_degrees(
    matrix: list[list[float]],
) -> list[float]:
    cosine_tilt = max(
        -1.0,
        min(1.0, matrix[2][2]),
    )

    tilt = math.acos(cosine_tilt)
    sine_tilt = math.sin(tilt)

    if abs(sine_tilt) < 1e-12:
        rotation = 0.0
        psi = math.atan2(
            matrix[1][0],
            matrix[0][0],
        )
    else:
        rotation = math.atan2(
            matrix[1][2],
            matrix[0][2],
        )

        psi = math.atan2(
            matrix[2][1],
            -matrix[2][0],
        )

    return [
        math.degrees(rotation),
        math.degrees(tilt),
        math.degrees(psi),
    ]


def sanitize_label(name: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        name,
    )

    return "_cs_" + cleaned


def convert(
    input_cs: Path,
    output_tsv: Path,
) -> None:
    dataset = Dataset.load(str(input_cs))

    try:
        fields = list(
            dataset.fields(
                exclude_uid=False,
            )
        )
    except TypeError:
        fields = list(dataset.fields())

        if "uid" not in fields:
            fields.insert(0, "uid")

    if "blob/path" not in fields:
        raise RuntimeError(
            "CryoSPARC .cs file does not contain blob/path"
        )

    skipped_fields = {
        "blob/path",
        "blob/idx",
        "blob/shape",
        "alignments3D/pose",
        "alignments3D/shift",
    }

    extra_fields = [
        field
        for field in fields
        if field not in skipped_fields
    ]

    extra_labels = [
        sanitize_label(field)
        for field in extra_fields
    ]

    base_directory = input_cs.resolve().parent

    output_tsv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_tsv.open(
        "w",
        encoding="utf-8",
    ) as output:
        headers = [
            "image_index",
            "image_path",
            "xmd_angle_rot",
            "xmd_angle_tilt",
            "xmd_angle_psi",
            "xmd_shift_x",
            "xmd_shift_y",
            *extra_labels,
        ]

        output.write(
            "\t".join(headers) + "\n"
        )

        for row in dataset.rows():
            image_path_text = scalar_value(
                row["blob/path"]
            )

            if not image_path_text:
                continue

            image_path = Path(image_path_text)

            if not image_path.is_absolute():
                image_path = (
                    base_directory /
                    image_path
                )

            image_index = ""

            if "blob/idx" in fields:
                image_index = str(
                    int(row["blob/idx"]) + 1
                )

            angles: list[float] | None = None

            if "alignments3D/pose" in fields:
                rotation_vector = vector_value(
                    row["alignments3D/pose"],
                    3,
                )

                if rotation_vector is not None:
                    rotation_matrix = (
                        rotation_vector_to_matrix(
                            rotation_vector
                        )
                    )

                    angles = matrix_to_zyz_degrees(
                        rotation_matrix
                    )

            shifts: list[float] | None = None

            if "alignments3D/shift" in fields:
                shifts = vector_value(
                    row["alignments3D/shift"],
                    2,
                )

            values = [
                image_index,
                str(image_path),
            ]

            if angles is None:
                values.extend(["", "", ""])
            else:
                values.extend(
                    str(value)
                    for value in angles
                )

            if shifts is None:
                values.extend(["", ""])
            else:
                values.extend(
                    str(value)
                    for value in shifts
                )

            for field in extra_fields:
                value = scalar_value(row[field])
                values.append(
                    value if value is not None else ""
                )

            output.write(
                "\t".join(values) + "\n"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CryoSPARC .cs particle dataset "
            "to an MDSPACE-compatible TSV file."
        )
    )

    parser.add_argument(
        "input_cs",
        type=Path,
        help="Input CryoSPARC .cs file",
    )

    parser.add_argument(
        "output_tsv",
        type=Path,
        help="Output TSV file",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    if not arguments.input_cs.is_file():
        print(
            (
                "cryosparc-cs-reader: "
                f"input file does not exist: "
                f"{arguments.input_cs}"
            ),
            file=sys.stderr,
        )

        return 1

    try:
        convert(
            arguments.input_cs,
            arguments.output_tsv,
        )

    except Exception as error:
        print(
            f"cryosparc-cs-reader: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
