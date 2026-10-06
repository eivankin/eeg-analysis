# EEG analysis — hyperventilation & speech study (stuttering / control)

Reproducible analysis of combined **EEG + GSR + breathing** recordings taken during a
fixed protocol (eyes-open/closed rest → hyperventilation → reading aloud → narrative
speech/interview → rest).  The point of interest is how a **person who stutters (PW)** and a
**control** respond to the hyperventilation challenge and to speaking/listening.

> ⚠️ **Design caveat:** this is effectively **n = 2 individuals** (one PW recorded on two
> different days, one control recorded as two crash-recovered parts).  Every "between-group"
> number is a single-subject observation, not a group statistic.

## Repo layout
```
analysis/            committed, reusable package (no hardcoded paths)
  dataset.py         EDF discovery + experimental/control (E/C) group parsing + stage overrides
  eegkit.py          general signal helpers (auto dead-channel detection, stages, bands)
notebooks/           runnable, self-contained notebooks (they REGENERATE every figure)
reports/             standalone report (EN + RU) as .tex + compiled .pdf
  figures/           PNG figures produced by the notebooks
```
The **raw data is never committed** (`data/` is git-ignored — large and personal).  The
exploratory `qc/` working scripts stay on disk but are also excluded from git; the notebooks
reproduce every figure from the committed `analysis` package alone, so there is **no hidden
dependency**.

## File naming scheme
```
<subject><E|C><session-id>[_p<part>].edf
   pwsE01.edf      subject "pws", group E (experimental — e.g. the stuttering group)
   ctlC02.edf      subject "ctl", group C (control)
   ctlC02_p2.edf   subject "ctl", group C, recording part 2
```
Files sharing the same alphabetic **subject prefix** are treated as the same person
(e.g. `pwsE01` + `pwsE02` = one person, two sessions — a natural test–retest).  A different
**group** letter means a different group.  Any recordings that follow this scheme work —
the code discovers them automatically and never hard-codes file names.

## Using your own data
Nothing is hard-coded to paths.  Set **one** of:
* `DATA_DIR = "..."` in the first cell of any notebook, or
* the environment variable `EEG_DATA_DIR`, or
* put your `.edf` files in `data/` next to this README (the default).

Electrode status, crash tails and stage boundaries are all **derived** from the data:
* dead channels = flat lines at the amplifier rail (auto-detected, incl. mid-session deaths/revivals);
* crash tails = a mass-flat block at the end (auto-detected and trimmed);
* stages = LSL event markers when present, otherwise an automatic estimate; a crash-recovered
  file without markers can use the transparent `dataset.STAGE_OVERRIDES` metadata (verify against the video).

## Environment
Tested in a conda env (`python 3.11`): `conda create -n eeg-qc python=3.11 && pip install -r requirements.txt`.

## Notebooks
| notebook | what it does | figures it writes to `reports/figures/` |
|---|---|---|
| `01_overview_qc` | discovery, electrode status + topoplots, dead-vs-active, aux channels, stage markers, alpha reactivity & PSD | `dead_topomaps`, `dead_vs_active`, `aux_ctrl_parts`, `psd_exp` / `psd_ctrl` |
| `02_hyperventilation` | % theta build-up during/after the challenge; post-challenge alpha | `hv_exp` |
| `03_speech_language` | interview vs rest, listening vs answering, Broca/Wernicke homologues, reading vs dialogue | `*_speak_listen` |
| `04_anomaly_screen` | spike/sharp-wave candidates (all blink/muscle), focal delta, line noise, posterior alpha asymmetry, crash tails | `spike_candidates` |

Open a notebook and **Run ▸ Restart ▸ Run All** (with data present).  Or from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Reports
`reports/report_en.{tex,pdf}` and `reports/report_ru.{tex,pdf}` are standalone (figures embedded).
To rebuild a PDF after regenerating figures: `cd reports && latexmk -xelatex report_en.tex`.

## Headline findings (see the reports for detail)
1. **Hyperventilation:** the same PW showed a strong theta build-up + *delayed* recovery on day 1
   but **no** build-up on day 2 — effort/compliance-dependent (breathing rate never rose; check video).
2. **Speech/interview:** frontal-midline theta is elevated in the PW **across the whole interview,
   including while only listening** — a dialogue-engagement state, not articulatory effort.  Control
   shows a small speaking-specific bump instead.
3. **Language homologues:** PW shows a right-shifted lateralization (left Broca relatively idle,
   right-dominant Wernicke) on both days; control is balanced — but identical while listening, so a
   tonic state.
4. **Reading vs dialogue:** same cortical effort signature despite different disfluency → the theta
   marker tracks *vocalizing*, not stuttering events.  Disfluency lives at the speech-motor level
   that 36-channel scalp EEG cannot resolve → **add audio + video-annotated stuttering events**.
5. **No epileptiform activity, no focal rhythmic delta, no 50 Hz line noise.**  A reproducible
   right-dominant posterior alpha asymmetry in the PW warrants an impedance check.
