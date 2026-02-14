# tocify

A simple personal RSS digester that triages feeds using local Ollama models.

## Prerequisites

1.  **Ollama**: Install [Ollama](https://ollama.com/) and ensure it is running.
2.  **Model**: Pull the default model (e.g., `ollama pull llama3.2`).
3.  **uv**: Install [uv](https://docs.astral.sh/uv/) for Python package management.

## Setup

1.  **Interests**: Define your reading preferences in `interests.md`.
2.  **Feeds**: Add your RSS feed URLs to `feeds.txt` (format: `Name | URL` or just `URL`).
3.  **.env**: (Optional) Create a `.env` file to customize settings:
    ```bash
    OLLAMA_MODEL=llama3.2
    WIKI_PATH=~/Code/matiswiki/
    ```

## Usage

Run the digester:

```bash
uv run digest.py
```

## How it works

The script fetches recent items from your feeds, filters them based on keywords in your `interests.md`, and then uses Ollama to score and summarize the top matches. The output is written to `digest.md` and synchronized to your digital garden if `WIKI_PATH` is configured.
