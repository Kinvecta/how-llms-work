# Security policy

## Status

This repository is course material for the video series **How Large Language Models Work: From Math to Transformers**. It is **not actively maintained**, has no supported versions and makes no security guarantees. Content and dependencies may be outdated, and dependencies may come to include known vulnerabilities. Review and scan anything you reuse.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead: go to the **Security** tab and choose **Report a vulnerability**, or open [a private report](https://github.com/Kinvecta/how-llms-work/security/advisories/new) directly. Reports are read on a best-effort basis, with no response or fix timeline. Content errors that are not security problems go through the [content error report](https://github.com/Kinvecta/how-llms-work/issues/new/choose) instead.

## What the repository does to lower risk

- No secrets, credentials, personal data, model weights, audio or video are committed. Every push and pull request is checked with a Gitleaks scan of the full history.
- The notebooks' Python dependencies are pinned in `requirements.txt` and audited with `pip-audit` on every push and weekly. Dependabot raises alerts and update PRs for pip and GitHub Actions.
- CodeQL scans the Python on every push and weekly.
- Third-party GitHub Actions are pinned to full commit SHAs, and workflows run with read-only tokens by default.

## Safe use of the notebooks

- The notebooks download public model weights (GPT-2 small) from the Hugging Face Hub. Run them in Colab or a virtual environment, not with elevated permissions.
- Model weights are loaded with `safetensors` or the `transformers` library from the official `openai-community/gpt2` repository. Don't swap in weights from a source you don't trust: some weight formats can execute code when loaded.
- The notebooks need no API keys and send no data anywhere except the downloads above.
