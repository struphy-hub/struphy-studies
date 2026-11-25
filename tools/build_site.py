#!/usr/bin/env python3
"""Generate the MkDocs pages of the site from the studies (``studies/<slug>/study.yaml``) into ``docs/``.

    python tools/build_site.py [--drafts] [--local-media [MEDIA_DIR]]
    mkdocs serve      # preview (or: mkdocs build)

    python tools/build_site.py --offline [--drafts]
        # generates the pages and builds the site into site/, which opens from disk (site/index.html), without a server

Drafts (``draft: true``) are only included with ``--drafts``. The media are embedded from the public share of each
study (MPCDF DataShare), see tools/study.py. With ``--local-media``, studies without a share link show their media
from the local media folder made by ``study.py create`` (a preview before the upload; such a build is never deployed).

The name of the site and the repository linked from the pages are ``site_name`` and ``repo_url`` of mkdocs.yml; the
text of the landing page (under the title) is ``intro.md``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import DEFAULT_MEDIA_DIR, STRUPHY_REPO, STUDIES, figure_path, media_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def mkdocs_setting(key: str) -> str:
    """A top-level string setting of mkdocs.yml (read as text: the file has tags that yaml.safe_load rejects)."""
    match = re.search(rf"(?m)^{key}:\s*(.+?)\s*$", (ROOT / "mkdocs.yml").read_text())
    if not match:
        sys.exit(f"mkdocs.yml: no {key!r}")
    return match.group(1).strip("\"'")


REPO = mkdocs_setting("repo_url").rstrip("/")
# offline build (--offline): pages are <slug>.html instead of <slug>/index.html, which changes relative media paths
OFFLINE = False


def short(commit: str | None) -> str:
    return commit[:9] if commit else "?"


_TAGS: dict[str, list[str]] | None = None


def struphy_tags() -> dict[str, list[str]]:
    """The tags of the Struphy repository by commit (annotated tags by the commit they point to), read once with
    ``git ls-remote``; empty if the repository cannot be reached (the pages are then built without tags)."""
    global _TAGS
    if _TAGS is None:
        _TAGS = {}
        try:
            out = subprocess.run(
                ["git", "ls-remote", "--tags", STRUPHY_REPO], capture_output=True, text=True, timeout=60, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:
            print(f"warning: tags of {STRUPHY_REPO} not available ({e}), commits are shown without tags")
            return _TAGS
        peeled = {}
        for line in out.splitlines():
            sha, _, ref = line.partition("\t")
            if not ref.startswith("refs/tags/"):
                continue
            name = ref.removeprefix("refs/tags/")
            if name.endswith("^{}"):
                peeled[name.removesuffix("^{}")] = sha
            else:
                peeled.setdefault(name, sha)
        for name, sha in peeled.items():
            _TAGS.setdefault(sha, []).append(name)
        for names in _TAGS.values():
            names.sort()
    return _TAGS


_TOKEN: str | None = None


def github_token() -> str:
    """Token for the GitHub API: $GITHUB_TOKEN or $GH_TOKEN (CI), else ``gh auth token``, else none (public only)."""
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        if not _TOKEN:
            try:
                out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20)
                _TOKEN = out.stdout.strip() if out.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                _TOKEN = ""
    return _TOKEN


def github_api(path: str):
    """GET ``https://api.github.com/<path>`` as JSON (None on any error)."""
    headers = {"Accept": "application/vnd.github+json"}
    if github_token():
        headers["Authorization"] = f"Bearer {github_token()}"
    try:
        request = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except Exception as e:  # noqa: BLE001 (no network, rate limit, private repository without token, ...)
        print(f"warning: GitHub API {path}: {e}")
        return None


def study_author(slug: str) -> str | None:
    """GitHub account of the author of a study: the one who opened the pull request that added its study.yaml, else
    (added without a pull request) the GitHub account of the author of that commit; None if not found (e.g. not
    committed yet, or offline)."""
    try:
        log = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", f"studies/{slug}/study.yaml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    if not log:
        return None
    if (
        subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        == "true"
    ):
        print("warning: shallow git clone, the authors of the studies are not shown (fetch the full history)")
        return None
    sha = log[-1]  # the first commit that added the file
    repo = REPO.removeprefix("https://github.com/")
    pulls = github_api(f"repos/{repo}/commits/{sha}/pulls") or []
    pulls = [p for p in pulls if p.get("merged_at")] or pulls
    if pulls:
        return min(pulls, key=lambda p: p["number"])["user"]["login"]
    commit = github_api(f"repos/{repo}/commits/{sha}") or {}
    return (commit.get("author") or {}).get("login")


def author_html(login: str | None) -> str:
    """The author line of a study page: avatar and link to the GitHub account."""
    if not login:
        return ""
    return (
        f'<p class="study-author"><img src="https://github.com/{login}.png?size=64" alt=""> '
        f'<a href="https://github.com/{login}">@{login}</a></p>\n'
    )


def commit_link(commit: str | None) -> str:
    """Link to the Struphy commit, followed by its tags (if any)."""
    if not commit:
        return "Struphy commit: not given"
    link = f"[`{short(commit)}`]({STRUPHY_REPO}/commit/{commit})"
    tags = [t for sha, names in struphy_tags().items() if sha.startswith(commit) for t in names]
    if tags:
        names = ", ".join(f"[`{t}`]({STRUPHY_REPO}/tree/{t})" for t in tags)
        link += f" (tag{'s' if len(tags) > 1 else ''} {names})"
    return link


def figure_html(url: str, text: str, description: str | None = None) -> str:
    """A figure with a visible caption, opened in the lightbox on click."""
    cap = html.escape(text)
    return (
        f'<figure class="study-figure"><a class="glightbox" href="{html.escape(url)}" '
        f'data-description="{html.escape(description or text)}"><img loading="lazy" src="{html.escape(url)}" '
        f'alt="{cap}"></a><figcaption>{cap}</figcaption></figure>'
    )


# share of a study whose media are previewed from the local media folder (--local-media), copied next to its page
LOCAL_SHARE = "local-preview"


def url(share: str, path: str, slug: str = "") -> str:
    """URL of the media file ``path`` of a study: on its public share, or (local preview) next to its page."""
    if share != LOCAL_SHARE:
        return media_url(share, path)
    # page URL studies/<slug>/ (served) or studies/<slug>.html (offline); the media are in docs/studies/<slug>/media/
    return f"{slug}/media/{path}" if OFFLINE else f"media/{path}"


def resolve_media(text: str, share: str | None, slug: str = "") -> str:
    """Replace the figures ``![caption](media:<path>)`` by figures with a visible caption, and other ``media:<path>``
    links by the URL of the file on the share."""
    if not text:
        return ""
    if not share:
        # without media: the caption of each figure as text
        return re.sub(r"!\[([^\]]*)\]\(media:[^)]*\)", r"*Figure: \1 (media not linked yet).*", text)
    text = re.sub(
        r"!\[([^\]]*)\]\(media:([^)\s]+)\)",
        lambda m: "\n" + figure_html(url(share, m.group(2), slug), m.group(1)) + "\n",
        text,
    )
    # Markdown links are resolved by MkDocs relative to the source file docs/studies/<slug>.md (not to the page URL
    # studies/<slug>/ like the HTML of the figures above)
    if share == LOCAL_SHARE:
        return re.sub(r"\(media:([^)\s]+)\)", lambda m: f"({slug}/media/{m.group(1)})", text)
    return re.sub(r"\(media:([^)\s]+)\)", lambda m: f"({url(share, m.group(1))})", text)


def caption(path: str) -> str:
    """Caption of a media file from its name, e.g. ``tae_omega_r.png`` -> ``tae omega r``."""
    return Path(path).stem.replace("_", " ")


def video_html(url: str, name: str) -> str:
    return (
        f'<figure class="study-video"><video controls preload="metadata" src="{html.escape(url)}"></video>'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
    )


def gallery(run: dict, share: str | None, slug: str) -> str:
    if not share:
        return '!!! warning "Media not linked yet"\n    The media of this study are not available.\n'
    out = []
    if run.get("videos"):
        videos = "\n".join(video_html(url(share, v, slug), caption(v)) for v in run["videos"])
        out.append(f'<div class="study-videos">\n{videos}\n</div>\n')
    if run.get("figures"):
        figures = "\n".join(
            figure_html(url(share, f, slug), caption(f), f"run {run['name']}: {caption(f)}") for f in run["figures"]
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
    lines = [f"# {title}\n", author_html(study.get("_author"))]
    meta = [f":octicons-calendar-16: {study.get('date', '')}", f":octicons-git-commit-16: {commit_link(commit)}"]
    if struphy.get("version"):
        meta.append(f"Struphy {struphy['version']}")
    if share == LOCAL_SHARE:
        meta.append(":octicons-file-media-16: media from the local media folder (preview)")
    elif share:
        meta.append(f":octicons-file-media-16: [media folder]({share})")
    meta.append(f":octicons-mark-github-16: [study folder]({REPO}/tree/main/studies/{slug})")
    if study.get("draft"):
        meta.append("**draft**")
    lines.append(" · ".join(meta) + "\n")
    lines.append(f"## Description\n\n{resolve_media(study.get('description', ''), share, slug)}\n")
    lines.append(f"## Main findings\n\n{resolve_media(study.get('findings', ''), share, slug)}\n")
    if study.get("figures") and share:
        figures = "\n".join(
            figure_html(
                url(share, figure_path(f), slug),
                f["caption"] if isinstance(f, dict) and f.get("caption") else caption(figure_path(f)),
            )
            for f in study["figures"]
        )
        lines.append("### Comparison figures\n")
        lines.append(f'<div class="study-figures">\n{figures}\n</div>\n')

    # galleries (videos and figures of each run)
    lines.append("## Videos\n")
    for run in study.get("runs") or []:
        lines.append(f'<a id="run-{run["name"]}"></a>\n')
        lines.append(f'??? abstract "{run["name"]}: {run.get("description", "")}"\n')
        lines.append(indent(gallery(run, share, slug)) + "\n")

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

    return "\n".join(lines)


def card(slug: str, study: dict) -> str:
    commit = (study.get("struphy") or {}).get("commit")
    thumb = f"assets/thumbnails/{slug}.png"
    title = study.get("title", slug)
    draft = " (draft)" if study.get("draft") else ""
    thumb_caption = study.get("thumbnail_caption")
    thumb_caption = f"\n\n    <small>*{thumb_caption}*</small>" if thumb_caption else ""
    login = study.get("_author")
    author = f":octicons-person-16: [@{login}](https://github.com/{login}) · " if login else ""
    picture = f"[![{title}]({thumb})](studies/{slug}.md){thumb_caption}\n\n    " if (DOCS / thumb).is_file() else ""
    return f"""-   {picture}**[{title}](studies/{slug}.md)**{draft}

    {author}:octicons-calendar-16: {study.get("date", "")} · :octicons-git-commit-16: {commit_link(commit)}

    {study.get("summary", "")}
"""


INDEX = """# {title}

![Struphy](assets/struphy_banner.png){{ .struphy-banner }}

{intro}

<div class="grid cards" markdown>

{cards}
</div>
"""


def intro() -> str:
    """Text of the landing page under the title: intro.md of the repository (Markdown)."""
    path = ROOT / "intro.md"
    return path.read_text().strip() if path.is_file() else ""


def main(argv=None) -> None:
    global OFFLINE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drafts", action="store_true", help="include draft studies")
    parser.add_argument(
        "--local-media",
        nargs="?",
        const=str(DEFAULT_MEDIA_DIR),
        default=None,
        metavar="MEDIA_DIR",
        help="preview: show the media of the studies without a share link from their local media folder "
        f"<MEDIA_DIR>/<slug>/ (default {DEFAULT_MEDIA_DIR}), copied into docs/; never deploy such a build",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="also build the site (mkdocs build --strict) into site/ so that it opens from disk (site/index.html) "
        "without a server, e.g. to download or send it as a zip",
    )
    args = parser.parse_args(argv)
    OFFLINE = args.offline

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
        local = Path(args.local_media) / slug if args.local_media else None
        if local is not None and not (study.get("media") or {}).get("share") and not local.is_dir():
            print(f"{slug}: no share link and no local media folder {local}; its media are not shown")
        elif local is not None and not (study.get("media") or {}).get("share"):
            # local preview of the media (not yet on DataShare): copied next to the page (docs/studies/<slug>/)
            shutil.copytree(local, DOCS / "studies" / slug / "media")
            study = {**study, "media": {**(study.get("media") or {}), "share": LOCAL_SHARE}}
            print(f"{slug}: media from {local} (local preview)")
        study = {**study, "_author": study_author(slug)}
        studies.append((slug, study))
        (DOCS / "studies" / f"{slug}.md").write_text(study_page(slug, study))
        thumb = path.parent / "thumbnail.png"
        if thumb.is_file():
            shutil.copy2(thumb, DOCS / "assets" / "thumbnails" / f"{slug}.png")
    # newest first
    studies.sort(key=lambda s: str(s[1].get("date", "")), reverse=True)
    cards = "\n".join(card(slug, study) for slug, study in studies) or "-   No studies yet.\n"
    (DOCS / "index.md").write_text(INDEX.format(title=mkdocs_setting("site_name"), intro=intro(), cards=cards))
    print(f"Generated {len(studies)} studies into {DOCS.relative_to(ROOT)}/")

    if args.offline:
        # the offline plugin of Material (enabled by OFFLINE=true in mkdocs.yml) makes the pages work from disk
        env = {**os.environ, "OFFLINE": "true"}
        subprocess.run(["mkdocs", "build", "--strict"], cwd=ROOT, env=env, check=True)
        print(f"Built the site into {(ROOT / 'site').relative_to(ROOT)}/: open site/index.html in a browser.")


if __name__ == "__main__":
    main()
