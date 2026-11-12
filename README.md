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
5. Preview: `python tools/build_site.py --drafts && mkdocs serve`.
6. Set `draft: false` and open a pull request. After merging, the site is rebuilt.

`python tools/study.py check --all` checks the media of all studies (done weekly by CI for the published ones).

## Reproducing a study

The study page gives the Struphy commit (`pip install "struphy @ git+https://github.com/struphy-hub/struphy@<commit>"`),
with its tags if it has any (looked up in the Struphy repository at each build; the site is rebuilt weekly, so a tag
added later shows up too), and, for each run, the params file and the commands. Struphy records its version and commit in the
`run_metadata.json` of each run.
