# Contributing

Thank you for helping. This project started in a jewellery workshop; the best contributions come from
people who use it on real files. Turkish and English are both welcome in issues and pull requests.

## Before you start

* Search the [issues](https://github.com/emrekaandrms/catalog-organizer/issues) first. For anything larger than a small fix, open an issue and
  describe what you want to change before writing code.
* **Never attach private CAD files, customer names, price lists or API keys.** If a bug needs a file to
  reproduce, make a small synthetic one (see `examples/make_demo_ring.py`) or describe the geometry.
* By contributing you agree your work is licensed under **GPL-3.0-or-later**, like the rest of the
  project.

## Development setup

```powershell
git clone https://github.com/emrekaandrms/catalog-organizer.git
cd catalog-organizer
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Tests that draw in the embedded Chromium need a GPU; skip them with `set CATALOG_ORGANIZER_NO_GPU=1`.
Tests that need private CAD files skip themselves when the files are absent.

## Guidelines

* **Tests come with the change.** A bug fix gets a regression test that fails without the fix (try it:
  undo the fix and watch the test fail). A feature gets tests for its rules.
* **Measure before you tune.** Thresholds in this code (stone-seat filters, weight corrections, render
  parameters) each exist because of a number measured on a real file. If you change one, say what you
  measured. Comments record *why*, not just *what*.
* **Say when something is an estimate.** The UI and the docs are honest about uncertainty; keep it so.
* **Do not mutate user data.** The app only reads CAD files.
* **Keep the engine switchable.** The web render engine is the default; the VTK renderer is the
  fallback. Changes to either should keep the other working.
* Python 3.11+, type hints on new code, `from __future__ import annotations`. Match the surrounding
  style; there is no formatter gate yet.
* Comments in the source sometimes refer to "design notes" or "the development log": those are the
  maintainers' private working notes and are not part of the repository. The comments are meant to be
  self-contained.

## Pull requests

1. Branch from `main`, keep the change focused, and write a clear description: what, why, how you
   checked it.
2. Run `pytest` (with `CATALOG_ORGANIZER_NO_GPU=1` if you have no GPU). CI runs the same.
3. Update the docs (`README.md`, `docs/`) and `CHANGELOG.md` when behaviour changes.

## Reporting bugs

Use the bug-report template. The most useful reports include the operating system, Python and package
versions (`pip list`), the exact steps, what you expected, and the message from **Logs** or the console.
For anything security-related, see [SECURITY.md](SECURITY.md) instead.

## Code of conduct

Be kind and assume good faith. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
