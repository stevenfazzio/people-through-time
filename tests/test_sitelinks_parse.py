import gzip

import pytest
from conftest import load_stage

fetch_sitelinks = load_stage("01_fetch_sitelinks.py")

ROWS = [
    "(1,42,'enwiki','Douglas Adams')",
    "(2,42,'dewiki','Douglas Adams')",
    "(3,42,'commonswiki','Category:Douglas Adams')",  # not a language Wikipedia
    "(4,42,'enwikiquote','Douglas Adams')",  # not a Wikipedia
    "(5,7,'enwiki','Someone else')",  # not on the roster
    "(6,1339,'frwiki','Jean-Sébastien Bach')",
    "(7,1339,'enwiki','O\\'Brien, test')",  # escaped quote and a comma inside the title
]


def write_dump(path, rows, row_per_line=True):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        if row_per_line:
            f.write("INSERT INTO `wb_items_per_site` VALUES\n" + ",\n".join(rows) + ";\n")
        else:
            f.write("INSERT INTO `wb_items_per_site` VALUES " + ",".join(rows) + ";\n")


def test_parse_sitelinks_keeps_roster_language_wikipedias(tmp_path):
    path = tmp_path / "wb.sql.gz"
    write_dump(path, ROWS)
    rows = fetch_sitelinks.parse_sitelinks(path, roster_ids={42, 1339})
    assert rows == [
        ("Q42", "enwiki", "Douglas_Adams"),
        ("Q42", "dewiki", "Douglas_Adams"),
        ("Q1339", "frwiki", "Jean-Sébastien_Bach"),
        ("Q1339", "enwiki", "O'Brien,_test"),
    ]


def test_old_one_line_layout_fails_loudly(tmp_path):
    # The fast parser reads one row per line. The old packed-INSERT layout must
    # raise (zero rows parsed), never silently return nothing.
    path = tmp_path / "wb.sql.gz"
    write_dump(path, ROWS, row_per_line=False)
    with pytest.raises(RuntimeError, match="no rows parsed"):
        fetch_sitelinks.parse_sitelinks(path, roster_ids={42})
