# CLAUDE.md

## Project

Jekyll + Minimal Mistakes (`remote_theme: mmistakes/minimal-mistakes`) personal
tech blog (`choeyunbeom.github.io`). AI-engineer job-search focused.

Home (`index.html`) = Featured Projects card grid + theme's Recent Posts.
`/portfolio/` (`_pages/portfolio.md`) = full project showcase + Reply competition card.

## Project showcase = `_includes/project-cards.html`

The Featured Projects / portfolio cards are **HTML + inline CSS**, not Markdown.
Both the home and the portfolio page render them with:

```liquid
{% include project-cards.html %}
```

Card grid is a self-contained `<style>` + `.pcard-grid` of `.pcard`s:
gradient icon header, **metric badges** (`.pcard-metric`), description, stack
line, and bottom-pinned `Code` / `Write-up` buttons. flexbox keeps card heights
even; grid is responsive (2-col desktop → 1-col mobile).

Rules when editing cards:
- Put numbers in **metric badges**, not bolded mid-sentence in the description.
- Keep project order = progression: arXiv RAG → FinScope → DefectVision → TORCS.
- Each card needs both a `Code` (GitHub) and `Write-up` (post) button.
- Don't edit theme CSS; keep styling inside the include's `<style>` block.

## Project repos (GitHub)

- arXiv RAG → `arxiv_rag_system`
- FinScope → `finscope`
- DefectVision → `defectvision`
- TORCS → `ibm_ai_race` (repo name is misleading; it IS the TORCS project)
- Reply challenge → `reply_ai_chal`

## Reply AI Challenge

Rank is always **137 / 1,971 (Top 7%)**. Never show the bare number without
context — it lives as a "Competition" card on `/portfolio/` only, NOT as a badge
on the home page (a context-free "137/1,971" reads as noise to recruiters).

## Conventions

- Post permalinks: `/:categories/:title/` (dateless, lowercase, spaces → `%20`).
  Never change an existing post's permalink — link to it from the portfolio.
- `_pages` get `layout: single`, `author_profile: true` by default (`_config.yml`).
- Nav lives in `_data/navigation.yml` (`main:`).
- CV: `assets/cv.pdf` (currently the MLE variant).

## Build

System Ruby is 2.6 and lacks the pinned bundler — use rbenv Ruby 3.2.2:

```bash
RB=~/.rbenv/versions/3.2.2/bin
rm -rf _site .jekyll-cache .jekyll-metadata   # avoid stale incremental cache
PATH="$RB:$PATH" $RB/bundle exec jekyll build
```

Always do a clean build before committing — incremental builds silently skip
include/layout changes.
