# Security policy

## Supported versions

This is beta software; only the latest commit on `main` is supported.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Use GitHub's private reporting:
**Security > Report a vulnerability** on the repository page. Include what you found, how to reproduce
it, and the impact you see. You can expect an acknowledgement within a few days.

## What to know about the application's data handling

* The app is **local-first**. It reads your CAD files and never modifies them.
* It contains **no telemetry**. The only network traffic is to the model provider you configure: Ollama on
  `127.0.0.1` by default, or a cloud provider (OpenAI, MiniMax, GLM, OpenCode Zen, an OpenAI-compatible
  endpoint) if you choose one. With a cloud provider, snapshot images of your pieces and listing text
  are sent to it.
* **API keys** are stored in the operating system's credential store (via `keyring`), not in `config/`
  or any other file in the project.
* The render engine runs an embedded Chromium and serves the 3D scenes on a **loopback-only** HTTP
  server (`127.0.0.1`, random port) for the app's own use.
* Do not commit `data/`, `cache/` or your own CAD files to a public fork. `.gitignore` excludes them.
