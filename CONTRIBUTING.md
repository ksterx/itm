# Contributing to It's My Turn (ITM)

ITM is a hobby research project. Contributions, issues, and discussions are welcome.

## Quick start

1. Fork and clone the repo
2. Set up the environment per [docs/implementation/environment.md](docs/implementation/environment.md)
3. Make your change on a feature branch
4. Run linters / tests locally
5. Open a Pull Request

## Code style

- Python 3.11+, modern type hints (`str | None`, `list[str]`)
- Pydantic v2 for any schema models
- `uv` for package management (`uv pip install`, `uv add`)
- `anyio` for async (avoid raw `asyncio` where possible)
- Format: `ruff format`
- Lint: `ruff check`

## Documentation

Documentation lives in `docs/` and is built with MkDocs Material. See [Documentation Policy](docs/meta/documentation-policy.md) for the full guide.

Key principles:

- Docs as code: edit `docs/*.md`, push, CI deploys to GitHub Pages
- Living docs: outdated info should be deleted or marked deprecated
- Write WHY, not WHAT — code already says what
- One source of truth — no duplicates

To preview locally:

```bash
uv pip install -e ".[docs]"  # or install mkdocs-material directly
mkdocs serve
# open http://localhost:8000
```

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` no functional change
- `test:` test changes
- `chore:` build/deps/tooling

Example: `docs: add label generation strategy for AMI dialog acts`

## Pull Requests

- Keep PRs focused. One concern per PR.
- Reference related issues with `Closes #N` or `Refs #N`.
- Update docs alongside code changes when relevant.

## License

By contributing, you agree that your contributions will be licensed under the BSD 2-Clause License.

## Code of Conduct

Be kind. Disagreements are welcome on technical merit; ad-hominem attacks are not.

## Questions?

Open an issue with the `question` label.
