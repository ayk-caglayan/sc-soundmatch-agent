## wiki/Makefile — Per-topic wiki management
##
## Standard Markdown links for GitLab; converts to wikilinks for Obsidian.
## For a unified multi-topic vault, use the top-level wiki/Makefile instead.
##
## Usage:
##   make obsidian    — export to ../wiki-obsidian/ with wikilinks
##   make import      — import from ../wiki-obsidian/ back (standard links)
##   make check       — find broken links and orphan pages
##   make stats       — show page/link/tag counts
##   make help        — show this help

SHELL      := /bin/bash
WIKI_DIR   := .
VAULT_DIR  := ../wiki-obsidian

.PHONY: help obsidian import check stats clean-vault

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

obsidian: ## Export wiki/ to ../wiki-obsidian/ with Obsidian wikilinks
	@echo "Exporting to $(VAULT_DIR)/ ..."
	@rm -rf $(VAULT_DIR)
	@mkdir -p $(VAULT_DIR)/sources
	@cp -a $(WIKI_DIR)/*.md $(VAULT_DIR)/
	@cp -a $(WIKI_DIR)/sources/*.md $(VAULT_DIR)/sources/ 2>/dev/null || true
	@python3 _convert.py to-wikilinks $(VAULT_DIR)
	@mkdir -p $(VAULT_DIR)/.obsidian
	@printf '{"useMarkdownLinks": false}\n' > $(VAULT_DIR)/.obsidian/app.json
	@echo "Done. Open $(VAULT_DIR)/ as an Obsidian vault."

import: ## Import from ../wiki-obsidian/ back to wiki/ (convert wikilinks to standard)
	@if [ ! -d "$(VAULT_DIR)" ]; then \
	  echo "*** $(VAULT_DIR)/ not found — nothing to import."; exit 1; \
	fi
	@echo "Importing from $(VAULT_DIR)/ ..."
	@cp -a $(VAULT_DIR)/*.md $(WIKI_DIR)/
	@cp -a $(VAULT_DIR)/sources/*.md $(WIKI_DIR)/sources/ 2>/dev/null || true
	@python3 _convert.py to-standard $(WIKI_DIR)
	@echo "Done. Wiki links converted to standard Markdown."

check: ## Find broken internal links and orphan pages
	@python3 _convert.py check $(WIKI_DIR)

stats: ## Show page, link, and tag counts
	@echo "=== Wiki Statistics ==="
	@printf "  Pages:          %d\n" $$(find $(WIKI_DIR) -name '*.md' -not -name 'Makefile' -not -path '*/\.*' | wc -l)
	@printf "  Source pages:   %d\n" $$(find $(WIKI_DIR)/sources -name '*.md' 2>/dev/null | wc -l)
	@printf "  Concept pages:  %d\n" $$(find $(WIKI_DIR) -maxdepth 1 -name '*.md' -not -name index.md -not -name overview.md -not -name log.md -not -name Makefile | wc -l)
	@printf "  Internal links: %d\n" $$(grep -roh '\[.*\](.*\.md)' $(WIKI_DIR)/*.md $(WIKI_DIR)/sources/*.md 2>/dev/null | wc -l)
	@printf "  Unique tags:    %d\n" $$(grep -rh '^tags:' $(WIKI_DIR)/*.md $(WIKI_DIR)/sources/*.md 2>/dev/null | tr ',' '\n' | sed 's/.*\[//;s/\].*//;s/^ *//;s/ *$$//' | sort -u | wc -l)

clean-vault: ## Remove the exported Obsidian vault
	rm -rf $(VAULT_DIR)
