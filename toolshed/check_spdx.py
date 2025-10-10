# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Any

import pathspec

# Intentionally puzzling together EXPECTED_SPDX_BYTES so that we don't overlook
# if the identifiers are missing in this file.
EXPECTED_SPDX_BYTES = (
    b"-".join((b"SPDX", b"License", b"Identifier: ")),
    b"-".join((b"SPDX", b"FileCopyrightText: ")),
)

SPDX_IGNORE_FILENAME = Path(".spdx-ignore")


def load_spdx_ignore() -> pathspec.PathSpec:
    if SPDX_IGNORE_FILENAME.exists():
        with SPDX_IGNORE_FILENAME.open(encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    lines.append(str(SPDX_IGNORE_FILENAME) + "\n")
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def has_spdx_or_is_empty(filepath: Path) -> bool:
    with filepath.open(mode="rb") as f:
        blob = f.read()

    if len(blob.strip()) == 0:
        return True

    good = True
    for expected_bytes in EXPECTED_SPDX_BYTES:
        if expected_bytes not in blob:
            print(f"MISSING {expected_bytes.decode()}{filepath!r}")  # noqa: T201
            good = False
    return good


def main(args: list[Any]) -> int:
    assert args, "filepaths expected to be passed from pre-commit"

    ignore_spec = load_spdx_ignore()

    returncode = 0
    for filepath in args:
        fp = Path(filepath)
        if ignore_spec.match_file(fp):
            continue

        if not has_spdx_or_is_empty(fp):
            returncode = 1

    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
