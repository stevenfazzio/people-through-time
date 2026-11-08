import gzip

from conftest import load_stage

fetch_redirects = load_stage("01b_fetch_redirects.py")


def write_dump(path, table, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(f"INSERT INTO `{table}` VALUES\n" + ",\n".join(rows) + ";\n")


def test_redirect_pages_are_ns0_redirects_only(tmp_path):
    path = tmp_path / "page.sql.gz"
    write_dump(path, "page", [
        "(10,0,'AccessibleComputing',1,0,0.85,'20261102','20261102',1,111,'wikitext',NULL)",
        "(12,0,'Anarchism',0,0,0.87,'20261107','20261107',1,114195,'wikitext',NULL)",  # article, not redirect
        "(13,2,'Some_user',1,0,0.1,'20261107','20261107',1,10,'wikitext',NULL)",  # user namespace
        "(14,0,'The_Rock\\'s_page',1,0,0.2,'20261107','20261107',1,10,'wikitext',NULL)",
        "(15,10,'0,_tricky',1,0,0.2,'20261107','20261107',1,10,'wikitext',NULL)",  # ns 10, title starts "0,"
    ])  # fmt: skip
    assert fetch_redirects.parse_redirect_pages(path) == {10: "AccessibleComputing", 14: "The_Rock's_page"}


def test_redirect_targets_keep_ns0_targets(tmp_path):
    path = tmp_path / "redirect.sql.gz"
    write_dump(path, "redirect", [
        "(10,0,'Computer_accessibility','','')",
        "(14,0,'Dwayne_Johnson','','')",
        "(20,14,'Some_category','','')",
    ])  # fmt: skip
    assert fetch_redirects.parse_redirect_targets(path) == {10: "Computer_accessibility", 14: "Dwayne_Johnson"}
