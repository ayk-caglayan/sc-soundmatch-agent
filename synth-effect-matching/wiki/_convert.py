#!/usr/bin/env python3
"""Convert between standard Markdown links and Obsidian wikilinks.

Usage:
    python3 _convert.py to-wikilinks <dir>   # [text](path.md) → [[target|text]]
    python3 _convert.py to-standard <dir>    # [[target|text]] → [text](path.md)
    python3 _convert.py check <dir>          # report broken links and orphans
"""

import glob
import os
import re
import sys


def find_md_files(directory: str) -> list[str]:
    """Return all .md files under directory."""
    return sorted(glob.glob(os.path.join(directory, "**", "*.md"), recursive=True))


# ---------------------------------------------------------------------------
# Standard → Wikilinks  (for Obsidian export)
# ---------------------------------------------------------------------------

def _bare_name(path: str) -> str:
    """'sources/hifigan.md' → 'hifigan', '../neural-audio-codecs.md' → 'neural-audio-codecs'."""
    return os.path.splitext(os.path.basename(path))[0]


def to_wikilinks(directory: str) -> None:
    """Convert [text](relative.md) links to [[name|text]] wikilinks."""
    for filepath in find_md_files(directory):
        with open(filepath) as f:
            content = f.read()

        def replace_standard(m: re.Match) -> str:
            text = m.group(1)
            target = m.group(2)
            # Only convert internal .md links
            if not target.endswith(".md"):
                return m.group(0)
            # Skip external URLs
            if target.startswith("http://") or target.startswith("https://"):
                return m.group(0)
            name = _bare_name(target)
            if text == name:
                return f"[[{name}]]"
            return f"[[{name}|{text}]]"

        new_content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_standard, content)
        if new_content != content:
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"  wikilinks: {filepath}")


# ---------------------------------------------------------------------------
# Wikilinks → Standard  (for GitLab import)
# ---------------------------------------------------------------------------

def _build_file_map(directory: str) -> dict[str, str]:
    """Map bare names to relative paths from wiki root.

    E.g. {'hifigan': 'sources/hifigan.md', 'overview': 'overview.md'}
    """
    fmap: dict[str, str] = {}
    for path in find_md_files(directory):
        rel = os.path.relpath(path, directory)
        bare = os.path.splitext(os.path.basename(path))[0]
        fmap[bare] = rel
    return fmap


def to_standard(directory: str) -> None:
    """Convert [[wikilinks]] to [text](relative-path.md) standard links."""
    fmap = _build_file_map(directory)

    for filepath in find_md_files(directory):
        with open(filepath) as f:
            content = f.read()
        if "[[" not in content:
            continue

        source_dir = os.path.dirname(filepath)

        def replace_wikilink(m: re.Match) -> str:
            inner = m.group(1)
            if "|" in inner:
                target, display = inner.split("|", 1)
            else:
                target = inner
                display = target
            target = target.strip()
            display = display.strip()

            # Handle wiki/ prefix
            if target.startswith("wiki/"):
                inner_path = target[5:]
                if not inner_path.endswith(".md"):
                    inner_path += ".md"
                abs_target = os.path.join(directory, inner_path)
            elif target in fmap:
                abs_target = os.path.join(directory, fmap[target])
            else:
                # Not found — assume wiki root
                abs_target = os.path.join(directory, target + ".md")

            rel_path = os.path.relpath(abs_target, source_dir)
            return f"[{display}]({rel_path})"

        new_content = re.sub(r"\[\[([^\]]+)\]\]", replace_wikilink, content)
        if new_content != content:
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"  standard: {filepath}")


# ---------------------------------------------------------------------------
# Link checker
# ---------------------------------------------------------------------------

def check_links(directory: str) -> None:
    """Report broken internal links and orphan pages."""
    all_files = {os.path.relpath(p, directory) for p in find_md_files(directory)}
    linked_targets: set[str] = set()
    broken: list[tuple[str, str, str]] = []

    for filepath in find_md_files(directory):
        source_dir = os.path.dirname(filepath)
        with open(filepath) as f:
            content = f.read()

        # Check standard links
        for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
            target = m.group(2)
            if target.startswith("http") or target.startswith("#"):
                continue
            # Strip any anchor
            target = target.split("#")[0]
            if not target:
                continue
            abs_path = os.path.normpath(os.path.join(source_dir, target))
            rel_from_root = os.path.relpath(abs_path, directory)
            linked_targets.add(rel_from_root)
            if not os.path.exists(abs_path):
                broken.append((
                    os.path.relpath(filepath, directory),
                    m.group(1),
                    target,
                ))

        # Check wikilinks (in case some remain)
        for m in re.finditer(r"\[\[([^\]]+)\]\]", content):
            broken.append((
                os.path.relpath(filepath, directory),
                "wikilink",
                m.group(1),
            ))

    # Orphans: pages not linked from any other page
    meta_pages = {"index.md", "log.md", "overview.md"}
    orphans = sorted(all_files - linked_targets - meta_pages)

    if broken:
        print(f"\n*** Broken links ({len(broken)}):")
        for src, text, target in sorted(broken):
            print(f"  {src}: [{text}]({target})")
    else:
        print("No broken links found.")

    if orphans:
        print(f"\n*** Orphan pages ({len(orphans)}) — not linked from any other page:")
        for o in orphans:
            print(f"  {o}")
    else:
        print("No orphan pages found.")

    # Summary
    total_links = sum(
        len(re.findall(r"\[[^\]]+\]\([^)]+\.md[^)]*\)", open(p).read()))
        for p in find_md_files(directory)
    )
    print(f"\nTotal: {len(all_files)} pages, {total_links} internal links, "
          f"{len(broken)} broken, {len(orphans)} orphans")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = {
    "to-wikilinks": to_wikilinks,
    "to-standard": to_standard,
    "check": check_links,
}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2])
