#!/usr/bin/env python3
"""Generate the MkDocs pages of the site from the studies (``studies/<slug>/study.yaml``) into ``docs/``.

    python tools/build_site.py [--drafts]
    mkdocs serve      # preview (or: mkdocs build)

Drafts (``draft: true``) are only included with ``--drafts``. The media are embedded from the public share of each
study (MPCDF DataShare), see tools/study.py.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import STRUPHY_REPO, STUDIES, media_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/struphy-hub/struphy-studies"


def short(commit: str | None) -> str:
    return commit[:9] if commit else "?"


def commit_link(commit: str | None) -> str:
    if not commit:
        return "Struphy commit: not given"
    return f"[`{short(commit)}`]({STRUPHY_REPO}/commit/{commit})"


def resolve_media(text: str, share: str | None) -> str:
    """Replace ``media:<path>`` (in Markdown links and images) by the URL of the file on the share."""
    if not text:
        return ""
    if not share:
        # without media: the caption of each figure as text
        return re.sub(r"!\[([^\]]*)\]\(media:[^)]*\)", r"*Figure: \1 (media not linked yet).*", text)
    return re.sub(r"\(media:([^)\s]+)\)", lambda m: f"({media_url(share, m.group(1))})", text)


def caption(path: str) -> str:
    """Caption of a media file from its name, e.g. ``tae_omega_r.png`` -> ``tae omega r``."""
    return Path(path).stem.replace("_", " ")


def video_html(url: str, name: str) -> str:
    return (
        f'<figure class="study-video"><video controls preload="metadata" src="{html.escape(url)}"></video>'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
    )


def gallery(run: dict, share: str | None) -> str:
    if not share:
        return '!!! warning "Media not linked yet"\n    The media of this study are not available.\n'
    out = []
    if run.get("videos"):
        videos = "\n".join(video_html(media_url(share, v), caption(v)) for v in run["videos"])
        out.append(f'<div class="study-videos">\n{videos}\n</div>\n')
    if run.get("figures"):
        figures = "\n".join(
            f'<a class="glightbox" href="{html.escape(media_url(share, f))}" data-description="{html.escape(caption(f))}">'
            f'<img loading="lazy" src="{html.escape(media_url(share, f))}" alt="{html.escape(caption(f))}" '
            f'title="{html.escape(caption(f))}"></a>'
            for f in run["figures"]
        )
        out.append(f'<div class="study-figures">\n{figures}\n</div>\n')
    return "\n".join(out)


def indent(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else "" for line in text.splitlines())


def study_page(slug: str, study: dict) -> str:
    share = (study.get("media") or {}).get("share")
    struphy = study.get("struphy") or {}
    commit = struphy.get("commit")
    title = study.get("title", slug)
    lines = [f"# {title}\n"]
    meta = [f":octicons-calendar-16: {study.get('date', '')}", f":octicons-git-commit-16: {commit_link(commit)}"]
    if struphy.get("version"):
        meta.append(f"Struphy {struphy['version']}")
    if share:
        meta.append(f":octicons-file-media-16: [media folder]({share})")
    meta.append(f":octicons-mark-github-16: [study folder]({REPO}/tree/main/studies/{slug})")
    if study.get("draft"):
        meta.append("**draft**")
    lines.append(" · ".join(meta) + "\n")
    lines.append(f"## Description\n\n{resolve_media(study.get('description', ''), share)}\n")
    lines.append(f"## Main findings\n\n{resolve_media(study.get('findings', ''), share)}\n")
    if study.get("figures") and share:
        figures = "\n".join(
            f'<a class="glightbox" href="{html.escape(media_url(share, f))}" data-description="{html.escape(caption(f))}">'
            f'<img loading="lazy" src="{html.escape(media_url(share, f))}" alt="{html.escape(caption(f))}"></a>'
            for f in study["figures"]
        )
        lines.append(f'<div class="study-figures">\n{figures}\n</div>\n')

    # reproduce
    lines.append("## Reproduce\n")
    if commit:
        lines.append(
            "Install the Struphy version of the study (in a fresh virtual environment) and compile the kernels:\n\n"
            f'```bash\npip install "struphy @ git+{STRUPHY_REPO}@{commit}"\nstruphy compile\n```\n'
        )
    else:
        lines.append("!!! note\n    The Struphy commit of this study is not given yet.\n")
    lines.append(
        f"The parameter and post-processing files are in [`studies/{slug}/params/`]({REPO}/tree/main/studies/{slug}/params). "
        "Run each command in that folder:\n"
    )
    lines.append("| run | description | params |\n|---|---|---|")
    for run in study.get("runs") or []:
        params = run.get("params")
        link = f"[`{params}`]({REPO}/blob/main/studies/{slug}/params/{params})" if params else ""
        lines.append(f"| [{run['name']}](#run-{run['name']}) | {run.get('description', '')} | {link} |")
    lines.append("")
    for run in study.get("runs") or []:
        lines.append(f"**{run['name']}**\n")
        lines.append(f"```bash\n{run.get('command', '')}\n{run.get('pproc', '')}\n```\n".replace("\n\n```", "\n```"))

    # galleries
    lines.append("## Runs\n")
    for run in study.get("runs") or []:
        lines.append(f'<a id="run-{run["name"]}"></a>\n')
        lines.append(f'??? abstract "{run["name"]}: {run.get("description", "")}"\n')
        lines.append(indent(gallery(run, share)) + "\n")
    return "\n".join(lines)


def card(slug: str, study: dict) -> str:
    commit = (study.get("struphy") or {}).get("commit")
    thumb = f"assets/thumbnails/{slug}.png"
    title = study.get("title", slug)
    draft = " (draft)" if study.get("draft") else ""
    return f"""-   [![{title}]({thumb})](studies/{slug}.md)

    **[{title}](studies/{slug}.md)**{draft}

    :octicons-calendar-16: {study.get("date", "")} · :octicons-git-commit-16: {commit_link(commit)}

    {study.get("summary", "")}
"""


INDEX = """# Struphy studies

![Struphy](assets/struphy_banner.png){{ .struphy-banner }}

Numerical studies with [Struphy]({struphy}): convergence studies, verifications and comparisons of models and
discretizations. Each study lists its runs with the parameter files and commands to reproduce them, and shows the
figures and videos of every run.

<div class="grid cards" markdown>

{cards}
</div>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drafts", action="store_true", help="include draft studies")
    args = parser.parse_args(argv)

    shutil.rmtree(DOCS, ignore_errors=True)
    (DOCS / "studies").mkdir(parents=True)
    (DOCS / "assets" / "thumbnails").mkdir(parents=True)
    shutil.copytree(ROOT / "site_assets", DOCS / "assets", dirs_exist_ok=True)

    studies = []
    for path in sorted(STUDIES.glob("*/study.yaml")):
        study = yaml.safe_load(path.read_text())
        if study.get("draft") and not args.drafts:
            continue
        slug = path.parent.name
        studies.append((slug, study))
        (DOCS / "studies" / f"{slug}.md").write_text(study_page(slug, study))
        thumb = path.parent / "thumbnail.png"
        if thumb.is_file():
            shutil.copy2(thumb, DOCS / "assets" / "thumbnails" / f"{slug}.png")
    # newest first
    studies.sort(key=lambda s: str(s[1].get("date", "")), reverse=True)
    cards = "\n".join(card(slug, study) for slug, study in studies) or "-   No studies yet.\n"
    (DOCS / "index.md").write_text(INDEX.format(struphy=STRUPHY_REPO, cards=cards))
    print(f"Generated {len(studies)} studies into {DOCS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
