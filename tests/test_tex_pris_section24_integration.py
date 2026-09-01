from __future__ import annotations

import re
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "tex" / "body.tex"
SI = ROOT / "tex" / "si_body.tex"
METHODS = ROOT / "tex" / "methods.tex"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _caption_for_label(tex: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    end = tex.index(marker)
    start = tex.rfind(r"\caption", 0, end)
    assert start >= 0
    return tex[start:end]


def _without_figure_environments(tex: str) -> str:
    return re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}",
        "",
        tex,
        flags=re.S,
    )


def _main_panel_sequence(tex: str, label: str) -> list[str]:
    sequence: list[str] = []
    pattern = re.compile(rf"\\ref\{{{re.escape(label)}\}}([a-f](?:--[a-f])?(?:,[a-f])?)")
    for match in pattern.finditer(_without_figure_environments(tex)):
        token = match.group(1)
        if "--" in token:
            start, end = token.split("--")
            panels = [chr(code) for code in range(ord(start), ord(end) + 1)]
        else:
            panels = token.split(",")
        for panel in panels:
            if panel not in sequence:
                sequence.append(panel)
    return sequence


def _section_24_prose_paragraphs() -> list[str]:
    body = _text(BODY)
    heading = (
        r"\subsection{PRIS screens candidates before expensive calculations and "
        r"explains five crystal contradictions}"
    )
    start = body.index(heading) + len(heading)
    end = body.index(r"\section{Discussion}", start)
    section = _without_figure_environments(body[start:end])
    section = re.sub(r"^\\label\{[^}]+\}(?:\\label\{[^}]+\})*\s*$", "", section, flags=re.M)
    return [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", section)
        if paragraph.strip()
    ]


def _paragraph_with(paragraphs: list[str], token: str) -> str:
    matches = [paragraph for paragraph in paragraphs if token in paragraph]
    assert len(matches) == 1, (token, len(matches))
    return matches[0]


def _methods_subsections() -> list[tuple[str, str]]:
    methods = _text(METHODS)
    matches = list(re.finditer(r"^\\subsection\{([^}]+)\}\s*$", methods, flags=re.M))
    subsections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(methods)
        subsections.append((match.group(1), methods[match.end():end].strip()))
    return subsections


def test_main_figure_map_is_closed_and_old_ranking_figure_is_removed():
    body = _text(BODY)
    assert r"figs/fig4_validation_synthesis.pdf" in body
    assert r"\label{fig:validation-synthesis}" in body
    assert r"figs/fig5_deployment.pdf" in body
    assert r"figs/fig4_validation.pdf" not in body
    assert r"figs/fig5_ranking.pdf" not in body
    assert r"figs/fig6_deployment.pdf" not in body


def test_results_keep_only_the_original_four_subsection_titles():
    body = _text(BODY)
    assert re.findall(r"^\\subsection\{([^}]+)\}", body, flags=re.M) == [
        "Autonomous agents discover eight laws through proposal, test and refutation",
        "PRIS balances experimental-structure satisfaction with damage detection",
        "Five complementary mechanisms turn screening into diagnosis",
        "PRIS screens candidates before expensive calculations and explains five crystal contradictions",
    ]
    assert not re.search(r"^\\(?:subsubsection|paragraph|subparagraph)\*?\{", body, flags=re.M)


def test_fig4_panels_are_first_cited_in_a_to_f_order():
    body = _text(BODY)
    tokens = [rf"\ref{{fig:validation-synthesis}}{panel}" for panel in "abcdef"]
    positions = [body.index(token) for token in tokens]
    assert positions == sorted(positions)


def test_every_main_figure_is_narrated_in_panel_order():
    body = _text(BODY)
    expected = {
        "fig:loop": list("abcde"),
        "fig:rules": list("abcd"),
        "fig:anatomy": list("abc"),
        "fig:validation-synthesis": list("abcdef"),
        "fig:deploy": list("abcdef"),
    }
    for label, panels in expected.items():
        assert _main_panel_sequence(body, label) == panels, label


def test_fig4_caption_is_descriptive_not_a_results_paragraph():
    caption = _caption_for_label(_text(BODY), "fig:validation-synthesis")
    for panel in "abcdef":
        assert rf"\textbf{{{panel}}}" in caption
    assert not re.search(r"\d+(?:\.\d+)?\\%", caption)
    assert not re.search(r"\b(?:outperform|detects?|screens?|retains?|gain|advantage)\b", caption, re.I)


def test_section24_places_the_unexpected_synthesis_conclusion_after_panel_d():
    body = _text(BODY)
    panel_c = body.index(r"\ref{fig:validation-synthesis}c")
    panel_d = body.index(r"\ref{fig:validation-synthesis}d")
    conclusion = body.lower().index("unexpected", panel_d)
    assert panel_c < panel_d < conclusion
    c_paragraph = body[body.rfind("\n\n", 0, panel_c):panel_c]
    assert re.search(r"PRIS-derived synthesis\s+score", c_paragraph)
    motivation = body[max(0, panel_c - 4500):panel_c].lower()
    assert "theoretical" in motivation
    assert "experiment" in motivation
    assert "synthesi" in motivation
    assert "Machine-learning materials design makes this extension necessary" not in body


def test_pss_formula_and_physicochemical_terms_are_shown_before_panel_c():
    body = _text(BODY)
    panel_c = body.index(r"\ref{fig:validation-synthesis}c")
    prefix = body[:panel_c]

    assert r"\begin{equation}" in prefix
    assert r"\mathrm{PSS}(\mathbf{x})" in prefix
    assert r"\Ssyn" not in prefix
    for coefficient in ("1.24", "0.84", "1.18", "4.90", "0.22", "0.59"):
        assert coefficient in prefix
    for abbreviation in (
        r"M_z",
        r"\eta_{\mathrm{site}}",
        r"\Delta_{\mathrm{BV}}",
        r"v_{\mathrm{atom}}",
        r"k_{\max}",
        r"f_{\mathrm{iso}}",
    ):
        assert abbreviation in prefix
    assert r"\widetilde x" in prefix
    assert r"\frac{x^\dagger-\mu_x}{\sigma_x}" in prefix
    assert "development-set medians" in prefix
    for mechanism in (
        "electrostatic",
        "crystallographic",
        "bond-valence",
        "packing",
        "coordination",
    ):
        assert mechanism in prefix.lower()


def test_pu_learning_lineage_and_expanded_pool_precede_the_pool_counts():
    body = _text(BODY)
    counts = body.index("364{,}592")
    context = body[max(0, counts - 1800):counts]

    assert r"\cite{jang2020structure" in context
    assert "CLscore" in context
    assert "CGCNN-PU" in context
    assert "MatterSim-1M-MLP-PU" in context
    assert re.search(r"expand(?:ed|ing)? the experimental\s+set", context, re.I)
    assert "unlabelled pool" in context
    assert "99{,}162" in context


def test_inverse_design_is_introduced_as_a_run_before_counts_and_is_diagnosed():
    body = _text(BODY)
    mattergen_count = body.index("1{,}081")
    context = body[max(0, mattergen_count - 900):mattergen_count]
    paragraph = _paragraph_with(
        _section_24_prose_paragraphs(), r"\ref{fig:validation-synthesis}f"
    )

    assert re.search(r"to test.{0,180}we (?:therefore )?ran MatterGen", context, re.I | re.S)
    assert "Property-conditioned inverse design makes this allocation problem sharper" not in body
    assert "61 of 61" in paragraph
    assert "D7" in paragraph
    assert "0.7-\\AA{}" in paragraph
    assert "distinct" in paragraph.lower()
    assert "volume per atom" in paragraph.lower()


def test_two_pu_encoders_and_their_reason_for_use_are_explicit():
    body = _text(BODY)
    assert "CGCNN-PU" in body
    assert "MatterSim-1M-MLP-PU" in body
    assert re.search(
        r"(?:pretrain(?:ed|ing).{0,100}(?:representation|embedding)|"
        r"(?:representation|embedding).{0,100}pretrain(?:ed|ing))",
        body,
        re.I | re.S,
    )
    assert re.search(r"(?:same|similar|consistent).{0,100}(?:trend|relation)", body, re.I | re.S)


def test_section24_each_fig4_paragraph_motivates_the_next_panel():
    paragraphs = _section_24_prose_paragraphs()
    expected_next_topics = {
        r"\ref{fig:validation-synthesis}a": ("mechanism",),
        r"\ref{fig:validation-synthesis}b": ("synthesi",),
        r"\ref{fig:validation-synthesis}c": ("model", "representation"),
        r"\ref{fig:validation-synthesis}d": ("rank", "priorit"),
        r"\ref{fig:validation-synthesis}e": ("inverse design", "generat"),
        r"\ref{fig:validation-synthesis}f": ("energy", "phonon", "experimental record"),
    }
    for token, topics in expected_next_topics.items():
        paragraph = _paragraph_with(paragraphs, token)
        tail = paragraph[-360:].lower()
        assert any(topic in tail for topic in topics), (token, tail)


def test_section24_makes_the_pris_pss_choice_and_cross_model_evidence_concrete():
    paragraphs = _section_24_prose_paragraphs()
    panel_c = _paragraph_with(paragraphs, r"\ref{fig:validation-synthesis}c")
    assert "conservative" in panel_c.lower()
    assert "tunable" in panel_c.lower()
    panel_d = _paragraph_with(paragraphs, r"\ref{fig:validation-synthesis}d")
    assert "ROC--AUC" in panel_d
    assert re.search(r"0\.9\d{2}", panel_d)
    assert "no synthesis label" in panel_d.lower()


def test_fig4f_has_the_same_narrative_weight_as_fig4c_and_fig4d():
    paragraphs = _section_24_prose_paragraphs()
    panel_c = _paragraph_with(paragraphs, r"\ref{fig:validation-synthesis}c")
    panel_d = _paragraph_with(paragraphs, r"\ref{fig:validation-synthesis}d")
    panel_f = _paragraph_with(paragraphs, r"\ref{fig:validation-synthesis}f")
    count = lambda paragraph: len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", paragraph))
    assert count(panel_f) >= 0.85 * min(count(panel_c), count(panel_d))
    assert "L1--L3" in panel_f
    assert "67.3\\%" in panel_f
    assert "maximum queue reduction of 67.3\\%" in panel_f
    assert "32.1\\%" in panel_f
    assert "140 of 140" in panel_f
    section = " ".join(paragraphs).lower()
    assert "we ran mattergen" in section
    assert "unique, unrelaxed structures" in section


def test_fig4_to_fig5_transition_moves_from_decisions_to_structural_causes():
    paragraphs = _section_24_prose_paragraphs()
    three_axis = next(
        paragraph
        for paragraph in paragraphs
        if "26{,}600" in paragraph and "phonon" in paragraph.lower()
    )
    tail = three_axis[-420:].lower()
    assert "structural variable" in tail
    assert "generated" in tail
    assert "fig.~\\ref{fig:deploy}" in tail


def test_section24_names_the_five_crystal_contradictions_in_order():
    paragraphs = _section_24_prose_paragraphs()
    section = " ".join(paragraphs).lower()
    markers = (
        "resolving the first contradiction",
        "resolving the second contradiction",
        "resolving the third contradiction",
        "resolving the fourth contradiction",
        "resolving the fifth contradiction",
    )
    positions = [section.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert not any(
        re.match(r"the (?:first|second|third|fourth|fifth) contradiction", paragraph, re.I)
        for paragraph in paragraphs
    )


def test_section24_each_fig5_paragraph_opens_the_next_mechanistic_test():
    paragraphs = _section_24_prose_paragraphs()
    expected_next_topics = {
        r"\ref{fig:deploy}a": ("gnome",),
        r"\ref{fig:deploy}b": ("strain", "controlled damage", "relaxation"),
        r"\ref{fig:deploy}c": ("raw generator", "mattergen"),
        r"\ref{fig:deploy}d": ("merge", "intervention"),
        r"\ref{fig:deploy}e": ("cost", "validation", "computational"),
    }
    for token, topics in expected_next_topics.items():
        paragraph = _paragraph_with(paragraphs, token)
        tail = paragraph[-360:].lower()
        assert any(topic in tail for topic in topics), (token, tail)

    merge = next(paragraph for paragraph in paragraphs if "113 contained" in paragraph)
    assert any(topic in merge[-360:].lower() for topic in ("occup", "chemical identity", "wrong element"))
    wrong_site = next(paragraph for paragraph in paragraphs if "cation--cation swaps" in paragraph)
    assert any(topic in wrong_site[-360:].lower() for topic in ("runtime", "cost", "scale", "routine"))


def test_methods_keep_five_compact_single_paragraph_subsections():
    subsections = _methods_subsections()
    assert [title for title, _ in subsections] == [
        "Data sets and study design",
        "Structural descriptors and rule-set evaluation",
        "Controlled damage and law selection",
        "External evaluation, synthesizability screening and statistics",
        "Autonomous agents, human oversight and reproducibility",
    ]
    for title, content in subsections:
        paragraphs = [part for part in re.split(r"\n\s*\n", content) if part.strip()]
        assert len(paragraphs) == 1, (title, len(paragraphs))
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", content)
        assert len(words) <= 220, (title, len(words))
    total_words = sum(
        len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", content))
        for _, content in subsections
    )
    assert total_words <= 850, total_words


def test_reader_visible_tex_uses_plain_screening_language():
    text = "\n".join((_text(BODY), _text(METHODS), _text(SI), _text(ROOT / "tex" / "front_body.tex")))
    assert not re.search(r"\btriag(?:e|es|ed|ing)\b", text, re.I)
    assert not re.search(r"\bprioriti[sz](?:e|es|ed|ing|ation)\b", text, re.I)
    assert "synthesizability screening" in text


def test_stability_score_branch_is_absent_from_main_and_si():
    text = "\n".join((_text(BODY), _text(METHODS), _text(SI), _text(ROOT / "tex" / "si.tex")))
    assert r"\Sstab" not in text
    assert "stability score" not in text.lower()
    assert not re.search(r"\bF[12]\b", text)
    assert "figS2_early_scores.pdf" not in text


def test_reader_visible_tex_contains_no_legacy_figure_number_filenames():
    text = "\n".join((_text(BODY), _text(METHODS), _text(SI)))
    for old_name in (
        "fig4_validation.pdf",
        "fig5_ranking.pdf",
        "fig6_deployment.pdf",
        r"fig5\_retractions.csv",
        r"fig6\_validity.csv",
        r"fig5\_twoway\_ladder.csv",
        r"fig6\_threeaxis.csv",
    ):
        assert old_name not in text


def test_new_si_figures_exist_and_are_cited_from_main_with_external_prefix():
    body = _text(BODY)
    si = _text(SI)
    labels = (
        "sifig:validation-boundary-omission",
        "sifig:polymorph-ranking",
        "sifig:l4-contribution",
        "sifig:pu-model-performance",
        "sifig:energy-phonon-record",
    )
    for label in labels:
        assert rf"\label{{{label}}}" in si
        assert rf"\ref{{si-{label}}}" in body


def test_every_main_and_si_figure_is_cited_outside_its_caption():
    body = _text(BODY)
    si = _text(SI)
    prose = _without_figure_environments(body + "\n" + si)
    labels = re.findall(r"\\label\{((?:fig|sifig):[^}]+)\}", body + "\n" + si)
    assert len(labels) == 24
    for label in labels:
        assert rf"\ref{{{label}}}" in prose or rf"\ref{{si-{label}}}" in prose, label


def test_all_figure_captions_follow_their_panel_order_and_contain_no_results_data():
    for path in (BODY, SI):
        tex = _text(path)
        for label in re.findall(r"\\label\{((?:fig|sifig):[^}]+)\}", tex):
            caption = _caption_for_label(tex, label)
            panels = re.findall(r"\\textbf\{([a-f])\}", caption)
            assert panels == sorted(panels), label
            assert len(panels) == len(set(panels)), label
            assert not re.search(r"\d+(?:\.\d+)?\\%", caption), label
            assert not re.search(
                r"\b(?:outperform|establish|reveal|demonstrate|resolve|explain|hidden|connects|supports)\b",
                caption,
                re.I,
            ), label


def test_all_figure_assets_are_in_the_numbered_common_folder():
    sources = _text(BODY) + "\n" + _text(SI)
    targets = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", sources)
    assert targets
    for target in targets:
        assert target.startswith("figs/fig"), target
        assert (ROOT / "tex" / target).is_file(), target


def test_figure_manifest_covers_every_tex_asset_once():
    manifest = json.loads(
        (ROOT / "tex" / "figure_scripts" / "figure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    finals = [entry["final"] for entry in manifest]
    assert len(finals) == len(set(finals)) == 24
    sources = _text(BODY) + "\n" + _text(SI)
    included = {
        Path(target).name
        for target in re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", sources
        )
    }
    assert included == set(finals)


def test_numbered_figure_folders_are_complete_and_hash_identical():
    manifest = json.loads(
        (ROOT / "tex" / "figure_scripts" / "figure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    finals = {entry["final"] for entry in manifest}
    latex_dir = ROOT / "tex" / "figs"
    review_dir = ROOT / "tex" / "key-file" / "final-numbered-pdfs"
    assert {path.name for path in latex_dir.glob("*.pdf")} == finals
    assert {path.name for path in review_dir.glob("*.pdf")} == finals
    for name in finals:
        assert (latex_dir / name).read_bytes() == (review_dir / name).read_bytes(), name


def test_reproducible_build_keeps_current_auxiliary_files_and_logs():
    build = _text(ROOT / "tex" / "build.sh")
    assert build.count("--keep-intermediates") == 3
    assert build.count("--keep-logs") == 3


def test_single_figure_entrypoint_exports_the_repository_import_path():
    generator = _text(ROOT / "tex" / "figure_scripts" / "generate_all.py")
    assert 'env["PYTHONPATH"]' in generator
    assert "subprocess.run(command, cwd=ROOT, env=env, stdout=sys.stderr, check=True)" in generator
