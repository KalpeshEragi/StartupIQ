from datetime import datetime, timezone

from startupiq.ingest.manifest import append_row, manifest_row, sha256_file


def test_sha256_known_value(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("abc", encoding="utf-8")
    assert sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_row_and_append(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("abc", encoding="utf-8")
    row = manifest_row(file, "https://example.com/a.txt", "test", datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
    assert row.startswith("| a.txt | https://example.com/a.txt | 2026-01-02 03:04 | 3 |")
    manifest = tmp_path / "MANIFEST.md"
    append_row(manifest, row)
    append_row(manifest, row)
    text = manifest.read_text(encoding="utf-8")
    assert text.count("| a.txt |") == 2
    assert text.startswith("# Raw data manifest")
