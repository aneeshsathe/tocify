# tocify — Weekly Journal ToC Digest (RSS → OpenAI → `digest.md`)

This repo runs a GitHub Action once a week (or on-demand) that:

1. pulls new items from a list of journal RSS feeds  
2. uses OpenAI to triage which items match your research interests  
3. writes a ranked digest to `digest.md` and commits it back to the repo

It’s meant to be forked and customized.

This was almost entirely vibe-coded as an exercise (I'm pleased at how well it works!)

---

## What’s in this repo

- **`digest.py`** — the pipeline (fetch RSS → filter → OpenAI triage → render markdown)
- **`feeds.txt`** — RSS feed list (supports comments; optionally supports `Name | URL`)
- **`interests.md`** — your keywords + narrative seed (used for relevance)
- **`prompt.txt`** — the prompt template (easy to tune without editing Python)
- **`digest.md`** — generated output (auto-updated)
- **`.github/workflows/weekly-digest.yml`** — scheduled GitHub Action runner
- **`requirements.txt`** — Python dependencies

---

## Quick start (fork + run)

### 1) Fork the repo
- Click **Fork** on GitHub to copy this repo into your account.

### 2) Enable OpenAI billing / credits
The OpenAI API requires an active billing setup or credits.
- Go to the OpenAI Platform and ensure billing is enabled and/or credits are available.
- If you see errors like `insufficient_quota` or `You exceeded your current quota`, this is the cause.
- I recommend putting in spending limits. This uses very little compute, but it's nice to be careful.

### 3) Create an OpenAI API key
Create an API key in the OpenAI Platform and copy it.

**Important:** never commit this key to the repo.

### 4) Add the API key as a GitHub Actions secret
In your forked repo:
- Go to **Settings → Secrets and variables → Actions**
- Click **New repository secret**
- Name: `OPENAI_API_KEY`
- Value: paste your OpenAI API key

That’s it—GitHub will inject it into the workflow at runtime.

### 5) Configure your feeds
Edit **`feeds.txt`**.

You can use comments:

```txt
# Core journals
Nature Neuroscience | https://www.nature.com/neuro.rss
PLOS Biology | https://journals.plos.org/plosbiology/rss

# Preprints
bioRxiv neuroscience | https://www.biorxiv.org/rss/subject/neuroscience.xml
