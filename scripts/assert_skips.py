"""Fail unless a pytest JUnit report skipped exactly the expected tests.

A skipped test and a deleted one look identical in a green summary, so a CI
job that only checks pytest's exit status can lose coverage without anyone
noticing. This compares the skipped test ids in the report against the ids
named on the command line, and fails on any difference in either direction:
an unexpected skip means the job measures less than it claims, and an
expected skip that ran means this list is stale.

    python scripts/assert_skips.py report.xml [EXPECTED_ID ...]

EXPECTED_ID is ``<classname>::<name>`` as JUnit records it, for example
``tests.test_lem_to_bem_unit::test_native_pure_grid_three_triangles_and_three_vertices``.
Name no ids to require that nothing skipped.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def skipped_ids(report: str) -> set[str]:
    root = ET.parse(report).getroot()
    ids = set()
    for case in root.iter("testcase"):
        if case.find("skipped") is not None:
            ids.add(f"{case.get('classname')}::{case.get('name')}")
    return ids


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    report, expected = argv[0], set(argv[1:])
    actual = skipped_ids(report)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    for test_id in unexpected:
        print(f"unexpected skip: {test_id}")
    for test_id in missing:
        print(f"expected skip did not happen (update the list): {test_id}")
    if unexpected or missing:
        return 1
    print(f"skips as expected: {len(actual)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
