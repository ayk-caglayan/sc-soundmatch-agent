# CLAUDE.md — synth-effect-matching Knowledge Base

Wiki structure based on the [LLM Knowledge Base Template](https://github.com/jeremyrayner/kb-template.git)
(clonable packaging of Andrej Karpathy's LLM knowledge base pattern).
This directory is both a paper collection and a wiki knowledge base.

## Structure

- **PDFs** in the root directory — the source papers
- **`wiki/`** — interlinked knowledge base (concept pages, source summaries, timeline)

## Ingest workflow (adding a new paper)

1. Place the PDF in this directory with its existing filename
2. Convert to text: `/Users/jos/miniforge3/bin/pdf2txt.py NewPaper.pdf > /tmp/newpaper.txt`
3. Read the text and discuss key takeaways
4. Write a source summary in `wiki/sources/` with YAML frontmatter:
   ```yaml
   ---
   tags: [relevant, tags]
   date: YYYY-MM-DD
   sources: 1
   ---
   ```
5. Update `wiki/index.md` — add the new source summary to the table
6. Update or create concept pages that the paper touches
7. Update `wiki/overview.md` if the paper shifts the big picture
8. Append an entry to `wiki/log.md`

## Wiki conventions

- Use **standard Markdown links** `[text](path.md)` (NOT Obsidian wikilinks)
- Cross-topic links use relative paths: `../../other-topic/wiki/page.md`
- Every wiki page has YAML frontmatter with `tags`, `date`, `sources`
- Note contradictions between papers; don't invent information; flag gaps

## Cross-topic awareness

This wiki is part of a larger knowledge base spanning multiple topic directories.
Related topics may include: ai-audio-codecs, attention, mamba, diffusion, ddsp, wavenet, etc.
When a concept is better covered in another topic's wiki, link to it rather than duplicating.
Use the path pattern: `../../<other-topic>/wiki/<page>.md`

## Important

- **Always convert PDFs to text first** — never use the Read tool on PDFs directly
- **Never modify PDFs** — they are immutable source documents
- **Always update `wiki/index.md`** after creating or modifying wiki pages
