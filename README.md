# Struphy studies

Numerical studies with [Struphy](https://github.com/struphy-hub/struphy), e.g. convergence studies, verifications
and comparisons of models and discretizations. The site is at **https://struphy-hub.github.io/struphy-studies/**.

Each study has a folder `studies/<yyyy-mm>-<name>/` with
- `study.yaml`: title, date, Struphy commit, descriptions, main findings, and the runs with their commands;
- `params/`: the parameter and post-processing files of the runs (and further scripts of the study);
- `thumbnail.png`: the picture on the landing page.

The figures and videos of the runs are **not in git**. They live in a folder on
[MPCDF DataShare](https://datashare.mpcdf.mpg.de), shared with a public link (without password), and the site embeds
them from there. To remove the media of a study, delete the folder or its link on DataShare.

## Add a study

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
source .venv/bin/activate

# 1. study folder and media folder from the Struphy output folders of the runs
python tools/study.py create 2026-11-my-study --title "My study" \
    --run base=/path/to/sim_base --run fine=/path/to/sim_fine \
    --figure /path/to/comparison.png --thumbnail /path/to/picture.png
```

`create` writes `studies/<slug>/study.yaml` (with TODOs), copies the params files, makes the thumbnail and puts the
figures and videos of the runs (from `<run>/results/`) into `~/struphy-studies-media/<slug>/`. Then:

1. Fill in the TODOs of `study.yaml`: summary (one sentence), description, findings, and the command of each run.
   Figures of the media folder can be used in the texts as `![caption](media:figures/comparison.png)`; the caption is
   shown under the figure, so say which runs it shows (and their line styles or colors). The figures of the whole
   study (`figures:`) can be given as `- path: figures/x.png` with a `caption:`. `thumbnail_caption` says what the
   thumbnail shows (e.g. the run). Replace the thumbnail with `python tools/study.py thumbnail <slug> <image>` (it is
   kept sharp: reduced only to 1600 px width).
2. Upload the media folder `~/struphy-studies-media/<slug>/` to DataShare (web interface or desktop client).
3. On DataShare, share this folder with a link, remove the password, and copy the link.
4. `python tools/study.py link <slug> <link>`: stores the link and checks that all media are there and served
   correctly.
5. Preview: `python tools/build_site.py --drafts && mkdocs serve`. Before the upload (steps 2-4), `--local-media`
   shows the media from the local media folder instead (never deploy such a build).
6. Set `draft: false` and open a pull request. After merging, the site is rebuilt.

`python tools/study.py check --all` checks the media of all studies (done weekly by CI for the published ones).

### Equations

Texts can contain LaTeX math: `$...$` inline and `$$...$$` for displayed equations, with a blank line before
the `$$`. Write texts with math as YAML block scalars (`description: |`), as `study.py create` does: in quoted
strings (`"..."`) a backslash starts an escape sequence. Math in figure captions is not rendered.

## Private studies

To share studies with chosen collaborators before (or instead of) publishing them here, create a private repository
from the template [struphy-studies-template](https://github.com/struphy-hub/struphy-studies-template). It has the same
tools and study format; its site is not deployed but downloaded by the collaborators (CI artifact) or built locally.
To publish a study from there, run in the private repository
```bash
python tools/study.py publish <slug> <path of a clone of this repository>
```
which copies the study folder here (with `draft: false`), and open a pull request. The tools of both repositories are
the same; change them in the template first and copy them here.

## Offline site

`python tools/build_site.py --offline` builds the site into `site/` so that it opens from disk (`site/index.html`),
without a server. The title and the text of the landing page are `site_name` in `mkdocs.yml` and `intro.md`.

## Reproducing a study

The study page gives the Struphy commit (`pip install "struphy @ git+https://github.com/struphy-hub/struphy@<commit>"`),
with its tags if it has any (looked up in the Struphy repository at each build; the site is rebuilt weekly, so a tag
added later shows up too), and, for each run, the params file and the commands. Struphy records its version and commit in the
`run_metadata.json` of each run.
