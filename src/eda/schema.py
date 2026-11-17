"""Feature-name schema for the anonymized Scania APS features.

Feature names look like ``<prefix>_<NNN>`` (e.g. ``ag_003``). The prefix
identifies an anonymized attribute group; for histogram attributes the numeric
suffix is the bin index. This module only reads *names*, never values, so it
cannot leak anything from the data.

Histogram groups are reported as CANDIDATES based on structure alone (a
contiguous run of exactly `bins` suffixes starting at 0). Checkpoint 2c
verifies them against the actual values.
"""

import re
from collections import defaultdict

FEATURE_NAME_PATTERN = re.compile(r"^([A-Za-z]+)_(\d+)$")
DEFAULT_HISTOGRAM_BINS = 10


def parse_feature_name(name):
    """Return (prefix, suffix_int) or None if the name does not fit the pattern."""

    match = FEATURE_NAME_PATTERN.match(str(name))
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def group_features(columns):
    """Group column names by prefix.

    Returns (groups, unparsed) where groups maps prefix -> columns ordered by
    numeric suffix, and unparsed lists names that do not fit the pattern.
    """

    buckets = defaultdict(list)
    unparsed = []

    for column in columns:
        parsed = parse_feature_name(column)
        if parsed is None:
            unparsed.append(column)
            continue
        prefix, suffix = parsed
        buckets[prefix].append((suffix, column))

    groups = {
        prefix: [column for _, column in sorted(members)]
        for prefix, members in sorted(buckets.items())
    }
    return groups, unparsed


def describe_groups(columns, bins=DEFAULT_HISTOGRAM_BINS):
    """Structural description of every feature group."""

    groups, unparsed = group_features(columns)
    described = {}

    for prefix, members in groups.items():
        suffixes = [parse_feature_name(column)[1] for column in members]
        contiguous_from_zero = suffixes == list(range(len(suffixes)))
        described[prefix] = {
            "n_features": len(members),
            "suffix_min": min(suffixes),
            "suffix_max": max(suffixes),
            "contiguous_from_zero": contiguous_from_zero,
            "histogram_candidate": contiguous_from_zero and len(members) == bins,
            "columns": members,
        }

    return {
        "n_groups": len(described),
        "groups": described,
        "histogram_candidate_prefixes": sorted(
            p for p, d in described.items() if d["histogram_candidate"]
        ),
        "unparsed_columns": unparsed,
    }
