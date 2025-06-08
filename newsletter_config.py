# --- Newsletter Configuration ---
# This file centralizes all configurable parameters for the newsletter generation pipeline.

# --- General Topic Configuration ---
TARGET_TOPIC_NAME = "AI in Software Development"  # Example: "Product Management", "Data Science", "Cybersecurity"
TARGET_TOPIC_KEYWORDS = [
    "artificial intelligence", "machine learning", "software engineering",
    "dev tools", "coding assistants", "generative AI", "code generation",
    "AI-assisted development", "natural language processing in code",
    "AI software lifecycle", "intelligent applications"
] # List of keywords defining the scope of the target topic. Used for LLM relevance checks.
NEWSLETTER_NAME = "AI Dev Today" # Name of the newsletter, used in email intros.

# --- Source-Specific Scraper Configuration ---

# Reddit Scraper Configuration
TARGET_SUBREDDIT = "artificialintelligence"  # Subreddit to scrape (without r/). E.g., "ProductManagement", "datascience"

# Long Form (Email) Scraper Configuration
# TODO: Update with actual sender emails for your newsletters
TARGET_EMAIL_SENDERS = [
    "noreply@exampleaitools.com",
    "updates@exampledevnews.org",
    "newsletter@deeplearningdigest.ai"
] # List of email senders for long-form content.

# Current News Scrapers (Hacker News, Techmeme, etc.) Configuration
# List of domains to exclude from general news aggregation.
# Modify this based on the TARGET_TOPIC to avoid irrelevant sources or include niche ones.
TARGET_TOPIC_EXCLUDED_DOMAINS = [
    'twitter.com', 'facebook.com', 'instagram.com', 'tiktok.com', # Social media often not primary sources
    'forbes.com', # Often paywalled or requires heavy interaction
    'seekingalpha.com', # Finance focused, might be noisy
    # General list from previous iteration, review based on topic:
    'mastodon.social', 'blueskyweb.xyz', 'reddit.com', # Reddit handled by its own scraper
    'searchenginejournal.com', 'mas.to', 'socialmediatoday.com',
    'cryptonews.com', 'techcrunch.com', 'theinformation.com', 'businessinsider.com',
    'indiatimes.com', 'threads.net', 'javascript:tgd','mastodon.online','siliconangle.com',
    'pocket-lint.com', 'androidpolice.com', "cointelegraph.com",'coindesk.com','bsky.app',
    'cryptobriefing.com','cryptopolitan.com','nngroup.com','thehill.com','bizjournals.com',
    'techdirt.com', 'theblock.co','nbcbayarea.com','usatoday.com', 'coinpedia.org',
    'unchainedcrypto.com','bitcoinist.com','wsj.com','pymnts.com','americanbanker.com',
    'collabora.com', 'barrons.com', 'france24.com','washingtonpost.com','latimes.com',
    'banger.show','awsdocsgpt','lightning.engineering','blockworks.co', 'sci-hub.se',
    'ibtimes.com', 'cnn.com','neowin.net','archive.org','journa.host','phonearena.com',
    'qz.com','lwn.net','appuals.com','inews.co','dealstreetasia.com','bankautomationnews.com',
    'gizchina.com', 'ethereumworldnews.com', 'livemint.com', 'cryptopotato.com',
    'decrypt.co','crypto.news','cryptoslate.com', 'bbcnews.com','ign.com','benshoof.org',
    'github.io', 'slashdot.org', 'nypost.com', 'nytimes.com', 'newsbtc.com',
    'substack.com', # Substack handled by email scraper if subscribed
    'pingwest.com', 'youtube.com', 'github.com', 'play.google.com', 'cnbc.com',
    'afb.org', 'techlusive.in'
]

# Product Hunt Scraper Configuration
ENABLE_PRODUCT_HUNT_SCRAPER = True # Set to False if Product Hunt is not relevant to the TARGET_TOPIC_NAME.
# Optional: PRODUCT_HUNT_TOPIC_PATH = "topics/developer-tools" # Specific Product Hunt topic page. Empty for homepage.

# --- LLM and Content Processing Configuration ---

# Summary Generator: Topic Filtering
# Topics to explicitly screen out as "irrelevant" by the LLM.
# Ensure this list doesn't conflict with your TARGET_TOPIC_NAME or TARGET_TOPIC_KEYWORDS.
FORBIDDEN_TOPICS = [
    "celebrity gossip", "sports results", "political campaigns unless directly impacting tech policy",
    "general stock market news not specific to tech companies", "automotive recalls",
    "space launches unless directly related to software/AI",
    "video game reviews unless focused on game development technology",
    "local community events"
]

# Summary Generator: Relevance Filtering Labels (Dynamically generated based on TARGET_TOPIC_NAME)
# These are used by the LLM to classify relevance and by the script to filter based on LLM response.
_RELEVANCE_LABEL_BASE = TARGET_TOPIC_NAME.lower().replace(" ", "_")
RELEVANCE_LABEL_POSITIVE = f"{_RELEVANCE_LABEL_BASE}_related"
RELEVANCE_LABEL_NEGATIVE = f"not_{_RELEVANCE_LABEL_BASE}_related"

# Summary Generator: Text length for embedding query in uniqueness check
MAX_TEXT_FOR_EMBEDDING_QUERY = 2000 # Max characters of article text to use for generating query embedding for uniqueness.

# OpenAI API Key Path
OPENAI_API_KEY_PATH = 'openai_keys.py' # Path to a file containing your OpenAI API key (e.g., `OPENAI_API_KEY="sk-..."`)
                                       # This file should be in .gitignore.

# --- Vector Database (ChromaDB) Configuration ---
VECTOR_DB_PATH = "./chroma_db_main_store" # Path to the directory where ChromaDB will store its data.

# Uniqueness and Continuation Logic
# SIMILARITY_THRESHOLD: For cosine distance (1 - similarity), a smaller value means more similar.
# E.g., a threshold of 0.2 means articles with a cosine distance < 0.2 (i.e., >80% similarity) are considered candidates for deduplication.
# This value is highly dependent on the embedding model used and needs careful tuning.
# Test with known similar/dissimilar articles to find a good balance.
SIMILARITY_THRESHOLD = 0.2  # Lower means MORE similar. (Range typically 0 to 1 for cosine distance)

# RECENCY_DAYS_FOR_UNIQUENESS: How many days back to check for general semantic repeats.
# Articles older than this but within SIMILARITY_THRESHOLD might still be flagged as similar.
RECENCY_DAYS_FOR_UNIQUENESS = 7 # Days

# ENABLE_CONTINUATION_CHECK: Whether to use an LLM to check if a similar recent article is a "significant development".
ENABLE_CONTINUATION_CHECK = True

# MAX_DAYS_FOR_CONTINUATION_CHECK: If ENABLE_CONTINUATION_CHECK is True, only apply the LLM check
# for articles published within this many days. Older similar articles are treated as simple repeats.
# This should be less than or equal to RECENCY_DAYS_FOR_UNIQUENESS.
MAX_DAYS_FOR_CONTINUATION_CHECK = 3 # Days
