#!/usr/bin/env python3
"""Manage the studies of struphy-studies.

    python tools/study.py create <slug> --run NAME=PATH [--run NAME=PATH ...] [options]
        Create ``studies/<slug>/`` (study.yaml, params, thumbnail) from Struphy output folders, and the media folder
        ``<media-dir>/<slug>/`` (figures and videos of the runs) to be uploaded to MPCDF DataShare.
    python tools/study.py link <slug> <public link of the uploaded media folder>
        Store the link in study.yaml and check that all media are there (see ``check``).
    python tools/study.py check [<slug> ... | --all]
        Check that the media of the studies are reachable through their links, with the right content types
        (and range requests for videos, which browsers need for seeking).

The media are not stored in git: they live in a folder on DataShare, shared with a public link (without password).
The site embeds them from ``https://<host>/public.php/dav/files/<token>/<path>``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
DEFAULT_MEDIA_DIR = Path.home() / "struphy-studies-media"
STRUPHY_REPO = "https://github.com/struphy-hub/struphy"
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm"}
# largest width of the thumbnail in pixels (the cards show it about 650 px wide, twice that on high-DPI screens)
THUMBNAIL_WIDTH = 1600
SLUG = re.compile(r"^\d{4}-\d{2}-[a-z0-9]+(-[a-z0-9]+)*$")


# ----------------------------------------------------------------------------------------------------------------------
# study.yaml
# ----------------------------------------------------------------------------------------------------------------------
def study_dir(slug: str) -> Path:
    return STUDIES / slug


def load_study(slug: str) -> dict:
    path = study_dir(slug) / "study.yaml"
    if not path.is_file():
        sys.exit(f"No study {slug!r} ({path} not found).")
    return yaml.safe_load(path.read_text())


def set_share(slug: str, share: str) -> None:
    """Write the link of the media share into study.yaml (only that line, so that comments and layout stay)."""
    path = study_dir(slug) / "study.yaml"
    text = path.read_text()
    new, n = re.subn(r"(?m)^(  share:).*$", rf"\1 {share}", text, count=1)
    if n != 1:
        sys.exit(f"{path}: no line '  share:' in the media section.")
    path.write_text(new)


def figure_path(entry) -> str:
    """Path of a figure entry of study.yaml: a path, or a dict with ``path`` and ``caption``."""
    return entry["path"] if isinstance(entry, dict) else entry


def media_files(study: dict) -> list[str]:
    """All media paths of a study (relative to its media folder), from the study figures and the runs."""
    files = [figure_path(f) for f in study.get("figures") or []]
    for run in study.get("runs") or []:
        files += list(run.get("videos") or []) + list(run.get("figures") or [])
    return files


def parse_share(share: str) -> tuple[str, str]:
    """``(host, token)`` of a DataShare (Nextcloud) public link ``https://<host>/s/<token>``."""
    url = urllib.parse.urlparse(share.strip())
    match = re.fullmatch(r"/s/([A-Za-z0-9]+)/?", url.path)
    if url.scheme != "https" or not match:
        sys.exit(f"Not a public link of the form https://<host>/s/<token>: {share}")
    return url.netloc, match.group(1)


def media_url(share: str, path: str) -> str:
    """Direct URL of the file ``path`` (relative to the shared folder) of a public share."""
    host, token = parse_share(share)
    return f"https://{host}/public.php/dav/files/{token}/" + urllib.parse.quote(path)


# ----------------------------------------------------------------------------------------------------------------------
# create
# ----------------------------------------------------------------------------------------------------------------------
def _read_metadata(run: Path) -> dict:
    path = run / "run_metadata.json"
    if not path.is_file():
        sys.exit(f"{run} is not a Struphy output folder (no run_metadata.json).")
    return json.loads(path.read_text())


def _code_info(meta: dict) -> dict:
    """Version and commit of Struphy recorded in run_metadata.json (newer Struphy versions), else empty."""
    info = meta.get("struphy") or {}
    return {k: info.get(k) for k in ("version", "git_commit", "git_dirty") if info.get(k) is not None}


def _command(meta: dict, params_name: str) -> str | None:
    """The command that started the run, from run_metadata.json (newer Struphy versions), else None."""
    argv = meta.get("command_line") or (meta.get("struphy") or {}).get("command_line")
    if not argv:
        return None
    args = [a for a in argv[1:]]
    ranks = meta.get("mpi_ranks") or 1
    prefix = f"mpirun -n {ranks} " if ranks > 1 else ""
    return prefix + "python " + " ".join([params_name] + args)


def _copy_unique(src: Path, dst_dir: Path, copied: dict) -> str:
    """Copy ``src`` into ``dst_dir`` once (by content); return its name there."""
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest in copied:
        return copied[digest]
    name = src.name
    k = 1
    while (dst_dir / name).exists():
        name = f"{src.stem}_{k}{src.suffix}"
        k += 1
    shutil.copy2(src, dst_dir / name)
    copied[digest] = name
    return name


def _thumbnail(src: Path, dst: Path) -> None:
    from PIL import Image

    with Image.open(src) as image:
        image = image.convert("RGB")
        if image.width > THUMBNAIL_WIDTH:  # only reduced, never enlarged (sharp on the cards)
            height = round(image.height * THUMBNAIL_WIDTH / image.width)
            image = image.resize((THUMBNAIL_WIDTH, height), Image.LANCZOS)
        image.save(dst, optimize=True)


def _yaml_list(items: list[str], indent: int) -> str:
    pad = " " * indent
    return "".join(f"\n{pad}- {item}" for item in items) if items else " []"


def create(args) -> None:
    slug = args.slug
    if not SLUG.match(slug):
        sys.exit(f"The slug must look like 2026-11-short-name (year-month-words, lower case): {slug!r}")
    target = study_dir(slug)
    if target.exists() and not args.force:
        sys.exit(f"{target} exists (use --force to overwrite).")
    media = Path(args.media_dir).expanduser() / slug
    if media.exists() and not args.force:
        sys.exit(f"{media} exists (use --force to overwrite).")
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(media, ignore_errors=True)
    (target / "params").mkdir(parents=True)
    media.mkdir(parents=True)

    runs, versions, commits, copied = [], set(), set(), {}
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"--run must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        run = Path(path).expanduser().resolve()
        meta = _read_metadata(run)
        code = _code_info(meta)
        if "version" in code:
            versions.add(code["version"])
        if "git_commit" in code:
            commits.add(code["git_commit"])
        params = Path(meta.get("params_path") or "")
        params_name = _copy_unique(params, target / "params", copied) if params.is_file() else None
        for extra in sorted(params.parent.glob("pproc_*.py")) if params.is_file() else []:
            _copy_unique(extra, target / "params", copied)

        results = run / args.results
        figures, videos = [], []
        if results.is_dir():
            for f in sorted(results.rglob("*")):
                if not f.is_file() or "frames" in f.relative_to(results).parts[:-1] and f.suffix in FIGURE_SUFFIXES:
                    continue
                if f.suffix in VIDEO_SUFFIXES:
                    kind, collection = "videos", videos
                elif f.suffix in FIGURE_SUFFIXES:
                    kind, collection = "figures", figures
                else:
                    continue
                rel = Path("runs") / name / kind / f.name
                (media / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, media / rel)
                collection.append(rel.as_posix())
        command = _command(meta, params_name) if params_name else None
        runs.append(
            dict(
                name=name,
                params=params_name,
                command=command,
                figures=figures,
                videos=videos,
                ranks=meta.get("mpi_ranks"),
                grid=(meta.get("grid") or {}).get("num_elements"),
            )
        )

    study_figures = []
    for f in args.figure or []:
        f = Path(f).expanduser()
        rel = Path("figures") / f.name
        (media / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, media / rel)
        study_figures.append(rel.as_posix())
    for f in args.script or []:
        _copy_unique(Path(f).expanduser(), target / "params", copied)

    thumbnail_src = Path(args.thumbnail).expanduser() if args.thumbnail else None
    if thumbnail_src is None:
        candidates = [media / f for f in study_figures] + [media / f for r in runs for f in r["figures"]]
        thumbnail_src = candidates[0] if candidates else None
    if thumbnail_src is not None:
        _thumbnail(thumbnail_src, target / "thumbnail.png")

    commit = args.commit or (commits.pop() if len(commits) == 1 else None)
    version = versions.pop() if len(versions) == 1 else None
    runs_yaml = ""
    for r in runs:
        runs_yaml += f"""
  - name: {r["name"]}
    description: TODO what distinguishes this run (grid {r["grid"]})
    params: {r["params"]}
    command: {json.dumps(r["command"] or "TODO the command that started the run")}
    pproc: TODO the post-processing command
    videos:{_yaml_list(r["videos"], 6)}
    figures:{_yaml_list(r["figures"], 6)}
"""
    text = f"""# Study of struphy-studies, see the README of the repository for the fields.
title: {json.dumps(args.title or slug)}
date: {args.date or datetime.date.today().isoformat()}
# drafts are not published on the site (preview them with: python tools/build_site.py --drafts)
draft: true
struphy:
  # the Struphy commit the runs were made with (pip install "struphy @ git+{STRUPHY_REPO}@<commit>")
  commit: {commit or "null"}
  version: {json.dumps(version) if version else "null"}
# one sentence, shown on the landing page
summary: TODO one sentence
description: |
  TODO the longer description (Markdown): the question, the setup, the model and the parameters.
findings: |
  TODO the main findings (Markdown). Figures of the media folder are written as ![caption](media:figures/<file>).
media:
  # the public link (without password) of the media folder on MPCDF DataShare, set by: python tools/study.py link
  share: null
figures:{_yaml_list(study_figures, 2)}
runs:{runs_yaml}"""
    (target / "study.yaml").write_text(text)

    n_files = sum(1 for f in media.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in media.rglob("*") if f.is_file()) / 1e6
    print(
        f"""
Created {target.relative_to(ROOT)}/ (study.yaml, params/, thumbnail.png).
Media of the study: {media} ({n_files} files, {size:.0f} MB).

Next steps:
1. Fill in the TODOs in {(target / "study.yaml").relative_to(ROOT)} (descriptions, findings, commands).
2. Upload the folder {media} to MPCDF DataShare (https://datashare.mpcdf.mpg.de):
   drag and drop it into the web interface, or use the desktop client.
3. In DataShare, share the uploaded folder {slug} (only this folder): Share > Share link,
   remove the password, and copy the link.
4. Run:   python tools/study.py link {slug} <copied link>
5. Preview the site:   python tools/build_site.py --drafts && mkdocs serve
6. Set draft: false and open a pull request.
"""
    )


# ----------------------------------------------------------------------------------------------------------------------
# link / check
# ----------------------------------------------------------------------------------------------------------------------
def list_share(share: str) -> set[str]:
    """All files of a public share (paths relative to the shared folder), via WebDAV PROPFIND."""
    host, token = parse_share(share)
    root = f"/public.php/dav/files/{token}/"
    request = urllib.request.Request(f"https://{host}{root}", method="PROPFIND", headers={"Depth": "infinity"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            tree = ET.fromstring(response.read())
    except urllib.error.HTTPError as error:
        sys.exit(f"Cannot list the share {share} (HTTP {error.code}); is the link public and without password?")
    files = set()
    for href in tree.iter("{DAV:}href"):
        path = urllib.parse.unquote(href.text or "")
        if path.startswith(root) and not path.endswith("/"):
            files.add(path[len(root) :])
    return files


def check_file(url: str, suffix: str) -> str | None:
    """None if ``url`` serves the file with the right content type (and range requests for videos), else a problem."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status, ctype = response.status, response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return f"not reachable ({error.reason})"
    expected = CONTENT_TYPES.get(suffix.lower())
    if expected and not ctype.startswith(expected):
        return f"content type {ctype!r}, expected {expected!r}"
    if suffix.lower() in VIDEO_SUFFIXES and status != 206:
        return f"no range requests (HTTP {status} instead of 206), seeking would not work"
    return None


def check_study(slug: str, share: str | None = None, quiet: bool = False) -> int:
    """Number of problems of the media of a study (missing, unexpected or wrongly served files)."""
    study = load_study(slug)
    share = share or (study.get("media") or {}).get("share")
    if not share:
        print(f"{slug}: no media link yet (python tools/study.py link {slug} <link>).")
        return 1
    expected = media_files(study)
    available = list_share(share)
    problems = [f"missing on the share: {f}" for f in expected if f not in available]
    extra = sorted(available - set(expected))
    for f in expected:
        if f in available:
            problem = check_file(media_url(share, f), Path(f).suffix)
            if problem:
                problems.append(f"{f}: {problem}")
    for problem in problems:
        print(f"{slug}: {problem}")
    if extra and not quiet:
        print(f"{slug}: {len(extra)} files on the share that study.yaml does not use (fine, not shown), e.g. {extra[0]}")
    if not problems:
        print(f"{slug}: all {len(expected)} media files OK.")
    return len(problems)


def link(args) -> None:
    load_study(args.slug)
    parse_share(args.share)
    n = check_study(args.slug, share=args.share)
    if n:
        sys.exit(f"{n} problems: the link was not stored. Fix the upload (or the paths in study.yaml) and run again.")
    set_share(args.slug, args.share.strip().rstrip("/"))
    print(f"Stored the link in studies/{args.slug}/study.yaml.")


def check(args) -> None:
    if args.all or args.published:
        slugs = sorted(p.name for p in STUDIES.iterdir() if (p / "study.yaml").is_file()) if STUDIES.is_dir() else []
        if args.published:
            slugs = [s for s in slugs if not load_study(s).get("draft")]
        if not slugs:
            print("No studies to check.")
            return
    else:
        slugs = args.slugs
    if not slugs:
        sys.exit("Give studies, --all or --published.")
    problems = sum(check_study(slug, quiet=True) for slug in slugs)
    sys.exit(1 if problems else 0)


def thumbnail(args) -> None:
    load_study(args.slug)
    _thumbnail(Path(args.image).expanduser(), study_dir(args.slug) / "thumbnail.png")
    print(f"Wrote studies/{args.slug}/thumbnail.png from {args.image}.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create a study from Struphy output folders")
    p.add_argument("slug", help="name of the study folder, e.g. 2026-11-lowmode-tae-shear-alfven")
    p.add_argument("--run", action="append", required=True, help="NAME=PATH of a Struphy output folder (repeat)")
    p.add_argument("--title", help="title of the study")
    p.add_argument("--date", help="date of the study (default: today), YYYY-MM-DD")
    p.add_argument("--commit", help="Struphy commit of the runs (default: from run_metadata.json, if recorded)")
    p.add_argument("--figure", action="append", help="a figure of the whole study (e.g. a comparison), repeat")
    p.add_argument("--script", action="append", help="a further script to keep with the params (repeat)")
    p.add_argument("--thumbnail", help="image for the thumbnail (default: the first study figure)")
    p.add_argument("--results", default="results", help="folder of the figures and videos in each run folder")
    p.add_argument("--media-dir", default=str(DEFAULT_MEDIA_DIR), help="where the media folder is created")
    p.add_argument("--force", action="store_true", help="overwrite an existing study and media folder")
    p.set_defaults(func=create)

    p = sub.add_parser("link", help="store and check the public link of the uploaded media folder")
    p.add_argument("slug")
    p.add_argument("share", help="public link https://datashare.mpcdf.mpg.de/s/<token> (without password)")
    p.set_defaults(func=link)

    p = sub.add_parser("thumbnail", help="replace the thumbnail of a study")
    p.add_argument("slug")
    p.add_argument("image", help="image file (reduced to at most THUMBNAIL_WIDTH pixels wide)")
    p.set_defaults(func=thumbnail)

    p = sub.add_parser("check", help="check the media of studies")
    p.add_argument("slugs", nargs="*")
    p.add_argument("--all", action="store_true", help="all studies")
    p.add_argument("--published", action="store_true", help="all studies that are not drafts")
    p.set_defaults(func=check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
