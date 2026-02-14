import os, re, json, time, math, hashlib, logging
from datetime import datetime, timezone, timedelta

import feedparser
import ollama
from dateutil import parser as dtparser
# from dateutil import parser as dtparser # Commented out if not needed, but keep if used
# import google.generativeai as genai 

from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# ---- config (env-tweakable) ----
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_ITEMS_PER_FEED = int(os.getenv("MAX_ITEMS_PER_FEED", "50"))
MAX_TOTAL_ITEMS = int(os.getenv("MAX_TOTAL_ITEMS", "100"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "3"))  # Restored to 3 days
INTERESTS_MAX_CHARS = int(os.getenv("INTERESTS_MAX_CHARS", "2000"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "400"))
PREFILTER_KEEP_TOP = int(os.getenv("PREFILTER_KEEP_TOP", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5")) # Keep small for local reliability
MIN_SCORE_READ = float(os.getenv("MIN_SCORE_READ", "0.20")) # Restored to 0.20
MAX_RETURNED = int(os.getenv("MAX_RETURNED", "15"))

# Use a simpler schema for prompt injection
SCHEMA_PROMPT = """
{
    "week_of": "YYYY-MM-DD",
    "notes": "Brief summary of trends",
    "ranked": [
        {
            "id": "item_id_string",
            "title": "Paper Title",
            "link": "URL",
            "source": "Source Name",
            "published_utc": "ISO8601 or null",
            "score": 0.0 to 1.0 (float),
            "why": "Reasoning for relevance",
            "tags": ["tag1", "tag2"]
        }
    ]
}
"""

# ---- tiny helpers ----
def load_feeds(path: str) -> list[dict]:
    feeds = []
    if not os.path.exists(path):
        logger.warning(f"{path} not found.")
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "|" in s:
                name, url = [x.strip() for x in s.split("|", 1)]
            else:
                name, url = None, s
            feeds.append({"name": name, "url": url})
    return feeds

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_prompt_template(path: str = "prompt.txt") -> str:
    if not os.path.exists(path):
        # Fallback template if file missing
        return """You are an expert research assistant for a Biophysics PhD.
Analyze the following RSS items based on the user's interests.

USER INTERESTS:
{{NARRATIVE}}
KEYWORDS: {{KEYWORDS}}

OUTPUT FORMAT:
Return valid JSON matching this structure:
{{SCHEMA}}

RSS ITEMS:
{{ITEMS}}
"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def section(md: str, heading: str) -> str:
    m = re.search(rf"(?im)^\s*#{1,6}\s+{re.escape(heading)}\s*$", md)
    if not m:
        return ""
    rest = md[m.end():]
    m2 = re.search(r"(?im)^\s*#{1,6}\s+\S", rest)
    return (rest[:m2.start()] if m2 else rest).strip()

def parse_interests_md(md: str) -> dict:
    keywords = []
    for line in section(md, "Keywords").splitlines():
        line = re.sub(r"^[\-\*\+]\s+", "", line.strip())
        if line:
            keywords.append(line)
    narrative = section(md, "Narrative").strip()
    if len(narrative) > INTERESTS_MAX_CHARS:
        narrative = narrative[:INTERESTS_MAX_CHARS] + "…"
    return {"keywords": keywords[:200], "narrative": narrative}


# ---- rss ----
def parse_date(entry) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    for key in ("published", "updated", "created"):
        val = entry.get(key)
        if val:
            try:
                dt = dtparser.parse(val)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return None

def fetch_rss_items(feeds: list[dict]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    items = []
    logger.info(f"Fetching {len(feeds)} feeds (lookback: {LOOKBACK_DAYS} days)...")
    
    for feed in feeds:
        url = feed["url"]
        try:
            d = feedparser.parse(url)
            # Priority: manual name > RSS title > URL
            source = (feed.get("name") or d.feed.get("title") or url).strip()
            
            for e in d.entries[:MAX_ITEMS_PER_FEED]:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if not (title and link):
                    continue
                dt = parse_date(e)
                if dt and dt < cutoff:
                    continue
                
                # Combine summary and description, maybe content
                summary_raw = (e.get("summary") or e.get("description") or "")
                if 'content' in e:
                     for c in e.content:
                         summary_raw += " " + c.value
                
                summary = re.sub(r"\s+", " ", summary_raw.strip())
                if len(summary) > SUMMARY_MAX_CHARS:
                    summary = summary[:SUMMARY_MAX_CHARS] + "…"
                
                items.append({
                    "id": sha1(f"{source}|{title}|{link}"),
                    "source": source,
                    "title": title,
                    "link": link,
                    "published_utc": dt.isoformat() if dt else None,
                    "summary": summary,
                })
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")

    # dedupe + newest first
    unique_items = {it["id"]: it for it in items}
    items = list(unique_items.values())
    items.sort(key=lambda x: x["published_utc"] or "", reverse=True)
    return items[:MAX_TOTAL_ITEMS]


# ---- local prefilter ----
def keyword_prefilter(items: list[dict], keywords: list[str], keep_top: int) -> list[dict]:
    if not keywords: 
        return items[:keep_top]
        
    kws = [k.lower() for k in keywords if k.strip()]
    def hits(it):
        text = (it.get("title","") + " " + it.get("summary","")).lower()
        return sum(1 for k in kws if k in text)
    
    scored = [(hits(it), it) for it in items]
    matched = [it for s, it in scored if s > 0]
    
    # If we don't have enough matches, fill with recent items
    if len(matched) < keep_top:
        needed = keep_top - len(matched)
        existing_ids = {it["id"] for it in matched}
        others = [it for it in items if it["id"] not in existing_ids]
        matched.extend(others[:needed])
        
    matched.sort(key=hits, reverse=True)
    return matched[:keep_top]


# ---- LLM triage (Ollama) ----
def call_llm_triage(interests: dict, items: list[dict]) -> dict:
    lean_items = [{
        "id": it["id"],
        "source": it["source"],
        "title": it["title"],
        "link": it["link"],
        "published_utc": it.get("published_utc"),
        "summary": (it.get("summary") or "")[:SUMMARY_MAX_CHARS],
    } for it in items]

    template = load_prompt_template()

    prompt = (
        template
        .replace("{{KEYWORDS}}", json.dumps(interests["keywords"], ensure_ascii=False))
        .replace("{{NARRATIVE}}", interests["narrative"])
        .replace("{{SCHEMA}}", SCHEMA_PROMPT.strip())
        .replace("{{ITEMS}}", json.dumps(lean_items, ensure_ascii=False))
    )

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"You are a research assistant. You MUST return ONLY valid JSON matching this schema:\n{SCHEMA_PROMPT}"},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={"temperature": 0.1}
            )
            
            content = response.get("message", {}).get("content", "")
            if not content:
                logger.warning(f"Empty content from Ollama on attempt {attempt+1}")
                continue
            
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    logger.error("Failed to get valid JSON from Ollama after retries.")
    return {"week_of": datetime.now().date().isoformat(), "notes": "Error processing batch.", "ranked": []}

def triage_in_batches(interests: dict, items: list[dict], batch_size: int) -> dict:
    week_of = datetime.now(timezone.utc).date().isoformat()
    total = math.ceil(len(items) / batch_size)
    all_ranked, notes_parts = [], []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.info(f"Triaging batch {i // batch_size + 1}/{total} ({len(batch)} items)")
        res = call_llm_triage(interests, batch)
        
        if res.get("notes", "").strip():
            notes_parts.append(res["notes"].strip())
        
        # Validate ranked items have required fields
        for r in res.get("ranked", []):
            if "id" in r and "score" in r:
                all_ranked.append(r)

    # Deduplicate and sort
    best = {}
    for r in all_ranked:
        rid = r["id"]
        if rid not in best or r["score"] > best[rid]["score"]:
            best[rid] = r

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return {"week_of": week_of, "notes": " | ".join(dict.fromkeys(notes_parts))[:1000], "ranked": ranked}


# ---- render ----
def render_digest_md(result: dict, items_by_id: dict[str, dict]) -> str:
    week_of = result["week_of"]
    notes = result.get("notes", "").strip()
    ranked = result.get("ranked", [])
    kept = [r for r in ranked if r["score"] >= MIN_SCORE_READ][:MAX_RETURNED]

    lines = [f"# Daily Reads ({week_of})", ""]
    if notes:
        lines += [f"**Summary:** {notes}", ""]
    
    lines += [
        f"**Selected:** {len(kept)} items (score ≥ {MIN_SCORE_READ:.2f})",
        "",
        "---",
        "",
    ]
    
    org_lines = [f"* Daily Reads ({week_of})"]
    if notes:
        org_lines += [f"  Summary: {notes}"]
    
    if not kept:
        return "\n".join(lines + ["_No relevant items found today._", ""])

    for r in kept:
        it = items_by_id.get(r["id"], {})
        tags = ", ".join(r.get("tags", [])) if r.get("tags") else ""
        summary = (it.get("summary") or "").strip()
        clean_summary = re.sub('<[^<]+?>', '', summary)
        
        # Markdown
        lines += [
            f"## [{r['title']}]({r['link']})",
            f"**Source:** {r['source']} | **Score:** {r['score']:.2f}",
            f"**Why:** {r['why'].strip()}",
            (f"**Tags:** {tags}" if tags else ""),
            "",
        ]
        if summary:
            lines += [f"> {clean_summary[:300]}...", ""]
        lines += ["---", ""]

        # Org Mode
        org_lines += [
            f"** TODO [[{r['link']}][{r['title']}]]",
            f"   SCHEDULED: <{datetime.now().strftime('%Y-%m-%d %a')}>",
            f"   - Source: {r['source']}",
            f"   - Score: {r['score']:.2f}",
            f"   - Why: {r['why'].strip()}",
            (f"   - Tags: {tags}" if tags else ""),
        ]
        if summary:
            org_lines += [
                "   #+BEGIN_QUOTE",
                f"   {clean_summary[:500]}...",
                "   #+END_QUOTE"
            ]

    # Write Org file too
    with open("digest.org", "w", encoding="utf-8") as f:
        f.write("\n".join(org_lines) + "\n")

    return "\n".join(lines)


def main():
    interests = parse_interests_md(read_text("interests.md"))
    feeds = load_feeds("feeds.txt")
    
    if not feeds:
        logger.error("No feeds found in feeds.txt")
        return

    items = fetch_rss_items(feeds)
    logger.info(f"Fetched {len(items)} RSS items (pre-filter)")

    today = datetime.now(timezone.utc).date().isoformat()
    if not items:
        logger.info("No items found.")
        return

    items = keyword_prefilter(items, interests["keywords"], keep_top=PREFILTER_KEEP_TOP)
    logger.info(f"Sending {len(items)} RSS items to Ollama (post-filter)")

    items_by_id = {it["id"]: it for it in items}

    result = triage_in_batches(interests, items, batch_size=BATCH_SIZE)
    md = render_digest_md(result, items_by_id)

    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Wrote digest.md")


if __name__ == "__main__":
    main()
