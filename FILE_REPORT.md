# File Report

This document summarises the purpose of each Python file in the project.

## src/current_news_scrapers.py
Scrapes URLs from a few curated sources like Hacker News, The Register and Techmeme. It fetches recent links, filters by domain and simple heuristics, then combines them into a list for further processing.

## src/long_form_scraper.py
Uses the Gmail API to pull newsletters from a dedicated inbox. It extracts subject lines and URLs from specified senders so that longer-form content can be included.

## src/scrape_articles.py
Checks which URLs allow scraping via `robots.txt` and downloads article text using Newspaper3k. It returns a dictionary mapping each URL to its title and text.

## src/email_generator.py
Collects content from several sources, formats it as HTML and sends an email using SendGrid. It also controls when the daily newsletter is sent.

## src/product_hunt_scraper.py
Scrapes the Product Hunt homepage for the top products of the day and returns their links, titles and descriptions.

## src/reddit_scraper.py
Uses the Reddit API to fetch top posts from r/ProductManagement from the last 24 hours.

## src/get_bcc_contacts.py
Retrieves all email addresses from a named SendGrid contact list so they can be BCC'd when sending the newsletter.

## src/summary_generator.py
Runs scraped articles through several GPT prompts: de-duplicating URLs, checking for relevance and generating one-sentence summaries. It also creates a short introduction for the email.

## src/daily_topic_newsletter.py
Generates a topic-focused newsletter each day. It stores embeddings of previously sent article summaries so that similar stories aren't repeated while still allowing coverage of new developments.
