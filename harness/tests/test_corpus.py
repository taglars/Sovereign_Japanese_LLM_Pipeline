"""Recovering the scripts from the rating form."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jv import corpus                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORM = ROOT / "data" / "rating_form.html"
KEY = ROOT / "data" / "blind_key.csv"

FAKE_FORM = """<html><body>
<script>
const DATA = [{"id": "S01", "html": "<section><h4>intro</h4><div class=\\"jp\\">\
&lt;p&gt;現在ご使用中の[型式]トラッ\
ク&lt;/p&gt;</div></section><section><h4>specs</h4><div class=\\"jp\\">\
&lt;ul&gt;&lt;li&gt;燃費が向上&lt;/li&gt;&lt;/ul&gt;\
</div></section>"}];
function go(){}
</script></body></html>"""


def _write(tmp: Path, text: str, name: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


def test_brackets_inside_script_text_do_not_end_the_array():
    """A "]" inside the Japanese text must not truncate the parse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write(Path(tmpdir), FAKE_FORM, "form.html")
        scripts = corpus.read_rating_form(path)
        assert len(scripts) == 1
        assert len(scripts[0].sections) == 2
        assert "[型式]" in scripts[0].sections[0][1]


def test_markup_is_stripped_but_seams_are_kept():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write(Path(tmpdir), FAKE_FORM, "form.html")
        rendered = corpus.read_rating_form(path)[0].render()
        assert rendered.count("<section label=") == 2
        assert "&lt;" not in rendered and "<p>" not in rendered
        assert 'label="intro"' in rendered


def test_a_form_without_the_array_is_refused():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write(Path(tmpdir), "<html>nothing here</html>", "form.html")
        try:
            corpus.read_rating_form(path)
        except ValueError as exc:
            assert "const DATA" in str(exc)
        else:
            raise AssertionError("a form with no DATA array should raise")


def test_blind_key_column_aliases():
    """The CSV says side/rec_type, the workbook sheet says arm/family."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        form = _write(tmp, FAKE_FORM, "form.html")
        scripts = corpus.read_rating_form(form)

        csv_style = _write(tmp, "script_id,side,vin,rec_type\nS01,qwen38,V1,light\n",
                           "a.csv")
        corpus.attach_blind_key(scripts, csv_style)
        assert (scripts[0].arm, scripts[0].family) == ("qwen38", "light")

        sheet_style = _write(tmp, "script_id,arm,vin,family\nS01,batch,V1,heavy\n",
                             "b.csv")
        corpus.attach_blind_key(scripts, sheet_style)
        assert (scripts[0].arm, scripts[0].family) == ("batch", "heavy")


def test_a_key_with_neither_alias_raises_rather_than_blanking():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        scripts = corpus.read_rating_form(_write(tmp, FAKE_FORM, "form.html"))
        bad = _write(tmp, "script_id,model,vin,kind\nS01,qwen38,V1,light\n", "c.csv")
        try:
            corpus.attach_blind_key(scripts, bad)
        except ValueError as exc:
            assert "arm" in str(exc)
        else:
            raise AssertionError("an unjoinable key must raise, not blank the arm")


def test_the_real_packet():
    """The actual 12 scripts, if the data files are present."""
    if not (FORM.exists() and KEY.exists()):
        print("    (skipped: data/rating_form.html not present)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "corpus.json"
        scripts = corpus.build(FORM, KEY, out)
        assert len(scripts) == 12
        assert len({s.vin for s in scripts}) == 12, "a vehicle appears twice"
        per_arm = {}
        for script in scripts:
            per_arm[script.arm] = per_arm.get(script.arm, 0) + 1
        assert set(per_arm.values()) == {3}, per_arm
        assert len(per_arm) == 4, per_arm
        for script in scripts:
            assert script.sections and script.chars > 1000
            rendered = script.render()
            assert "&lt;" not in rendered
        # It round-trips.
        again = corpus.load(out)
        assert [s.script_id for s in again] == [s.script_id for s in scripts]
        assert again[0].render() == scripts[0].render()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:                              # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("corpus: " + ("all checks pass" if not failures else f"{failures} FAILED"))
    raise SystemExit(1 if failures else 0)
