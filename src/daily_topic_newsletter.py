"""Daily topic newsletter generator.

This module scrapes articles from the existing scrapers, filters them
for a given topic and sends a daily email.  It stores embeddings of
previously sent summaries so that we avoid sending the same story
repeatedly, while still allowing follow ups on the same topic if new
developments occur.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import openai
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from scrape_articles import scrape_articles

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SENDGRID_API_KEY:
    raise ValueError("SENDGRID_API_KEY environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY


@dataclass
class HistoryItem:
    url: str
    embedding: List[float]
    date: str


@dataclass
class DailyTopicNewsletter:
    topic: str
    embed_store: Path = field(init=False)
    history: List[HistoryItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.embed_store = Path(f"embeddings_{self.topic.lower()}.json")
        self.history = self._load_history()

    # ------------------------------------------------------------------
    def _load_history(self) -> List[HistoryItem]:
        if self.embed_store.exists():
            with open(self.embed_store, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [HistoryItem(**item) for item in raw]
        return []

    # ------------------------------------------------------------------
    def _save_history(self) -> None:
        data = [item.__dict__ for item in self.history[-100:]]
        with open(self.embed_store, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> List[float]:
        resp = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return resp["data"][0]["embedding"]

    # ------------------------------------------------------------------
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarise the following text:\n{text}"}],
            max_tokens=150,
            temperature=0.7,
        )
        return resp["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    def _filter_topic(self, text: str) -> bool:
        prompt = f"Answer with 'yes' or 'no'. Is the following text about {self.topic}?\n{text}"
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0.0,
        )
        answer = resp["choices"][0]["message"]["content"].strip().lower()
        return answer.startswith("yes")

    # ------------------------------------------------------------------
    def _select_new_articles(self, articles: Dict[str, Dict]) -> Dict[str, Dict]:
        selected: Dict[str, Dict] = {}
        now = datetime.utcnow()

        for url, data in articles.items():
            if not self._filter_topic(data["text"]):
                continue
            summary = self._summarise(data["text"])
            emb = self._embed_text(summary)

            duplicate = False
            for item in self.history:
                sim = self._cosine_similarity(emb, item.embedding)
                days_old = now - datetime.fromisoformat(item.date)
                if sim > 0.85 and days_old < timedelta(days=5):
                    duplicate = True
                    break
            if duplicate:
                continue

            selected[url] = {"title": data["title"], "summary": summary}
            self.history.append(HistoryItem(url=url, embedding=emb, date=now.isoformat()))

        self._save_history()
        return selected

    # ------------------------------------------------------------------
    def _format_email(self, articles: Dict[str, Dict]) -> str:
        html = f"<h2>Daily {self.topic} News</h2>"
        for art in articles.values():
            html += f"<h4>{art['title']}</h4><p>{art['summary']}</p>"
        return html

    # ------------------------------------------------------------------
    def _send_email(self, html: str) -> None:
        message = Mail(
            from_email="news@example.com",
            to_emails="you@example.com",
            subject=f"Daily {self.topic} Newsletter",
            html_content=html,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)

    # ------------------------------------------------------------------
    def run_once(self) -> None:
        articles = scrape_articles()
        selected = self._select_new_articles(articles)
        if not selected:
            print("No new articles today")
            return
        html = self._format_email(selected)
        self._send_email(html)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    TOPIC = os.getenv("NEWS_TOPIC", "AI")
    newsletter = DailyTopicNewsletter(TOPIC)
    while True:
        newsletter.run_once()
        time.sleep(24 * 60 * 60)
