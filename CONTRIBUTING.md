# Contributing

Thanks for your interest. A few things to know before you open anything.

## This repository is not actively maintained

It is a snapshot of course material that accompanies a recorded video series. Episodes are added as they are released, but there is no roadmap, no release schedule and no support. Issues and pull requests are read on a best-effort basis and may not get a response. The videos cannot change, so the slides and transcripts stay as recorded.

## Reporting an error in the content

If a number, equation, slide, transcript or notebook cell is wrong, open a [content error report](https://github.com/Kinvecta/how-llms-work/issues/new/choose). Say which file and where, what it says, what it should say, and a source if you have one. If a notebook `assert` fails, paste the output and your library versions (`pip freeze | grep -E "torch|transformers|numpy"`): a library update can shift a probability in the last decimal place.

Small fixes to the Markdown files (typos, broken links, outdated URLs) are welcome as pull requests. Keep them minimal and one topic per PR.

## Reporting a security problem

Do not open a public issue. Follow [SECURITY.md](SECURITY.md) and use GitHub's private vulnerability reporting.

## Changing or extending the material

Fork the repository. The licenses allow it:

- Course content (slides, transcripts, episode pages, research briefs, notebook text): [CC BY 4.0](LICENSE-CONTENT.md). Adapt and share with attribution to Mark D. Espinoza / Kinvecta, and note what you changed.
- Code (notebook code cells, `tools/`): [MIT](LICENSE).

New episodes, translations or a rewrite for your own class belong in your fork, not here.

## What not to commit

- No secrets, API keys, tokens or credentials.
- No personal data.
- No model weights, audio or video files; the notebooks download weights and the repository links to the videos.
- No notebook outputs with local file paths in them.

## Running the checks locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tools/check_notebooks.py --fast   # the numpy-only notebooks, in seconds
python tools/check_notebooks.py          # everything, including the GPT-2 notebooks
ruff check tools/
```

Workflows pin third-party actions to full commit SHAs. If you update one, update the version comment next to it.
