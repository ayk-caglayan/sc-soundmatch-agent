# Repository Guidelines

## Project Structure & Module Organization
The repository centers on curated research for synthesizer/effect parameter matching. PDF sources (`SynthMatch-*.pdf`, `DeepAFx21.pdf`, etc.) live at the root so they stay easy to cite. `README.md` is the canonical, chronological index and should always reference every PDF that ships with the repo. Supplementary research notes belong in `Notes.md` (polished) or `Notes-out.md` (scratch). `References.bib` holds BibTeX snippets for papers that need formal citations. Agent settings live under `.claude/` and rarely change.

## Build, Test, and Development Commands
- `ls -1 SynthMatch-*.pdf | sort` — quick audit that new PDFs follow the `SynthMatch-Topic-YYYY.pdf` pattern and did not land in nested folders.
- `rg -n "SynthMatch" README.md` — verify that newly added titles already exist (avoids duplicates) or locate the line to update.
- `markdownlint README.md AGENTS.md Notes*.md` — catch heading, spacing, and spelling issues before opening a PR.

## Coding Style & Naming Conventions
- Markdown bullets follow `- [YYYY-MM - Title](Filename.pdf)`; keep the date prefix four digits for year and two for month so sorting stays lexical.
- When linking external code, mirror the format already in the README: inline URL plus optional `(code)` link.
- Keep prose wrap at ~100 characters, use sentence case headings, and prefer descriptive filenames such as `SynthMatch-ModulationDiscovery-DDSP-2510.06204v1.pdf`.

## Testing Guidelines
- There is no automated test suite; treat manual verification as your “test.” Confirm every new link renders by opening the PDF locally (`open SynthMatch-Example.pdf` on macOS).
- Re-run `markdownlint` after edits and confirm `git status` only shows intended changes (no inadvertently touched PDFs).
- For citation updates, paste the BibTeX through a validator such as `bibtool -q -s References.bib` if you have it installed.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative commits such as `Add CLAUDE.md documentation`, `Add PDF files for Synth Matching`, or `master: nail down missing dates`.
- Each PR should state the intent (e.g., “Add 2025 modulation discovery paper”), enumerate added PDFs, and mention any README ordering changes.
- Link to related issues or discussion threads when applicable, and attach screenshots only if you changed rendered Markdown formatting.

## Agent-Specific Tips
- Work from the repo root, keep relative links intact, avoid renaming historical PDFs unless coordinated, and stage only the Markdown/PDF additions you intend to ship.
