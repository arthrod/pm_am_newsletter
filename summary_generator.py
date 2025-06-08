import openai
import time
import random
from scrape_articles import scrape_articles
import datetime # Import the module itself
import os 
import re # For cleaning collection name
import numpy as np
from embedding_utils import load_embedding_model, get_embedding
from vector_db_utils import init_chroma_client, get_or_create_collection, query_similar_articles
import newsletter_config as config

# Configuration values are now sourced from newsletter_config.py
# Example: config.VECTOR_DB_PATH, config.TARGET_TOPIC_NAME, etc.

# TODO: Crucial cross-file dependency: email_generator.py MUST store 'summary' in ChromaDB metadata for this to work (already handled in email_generator.py).


openai.api_key_path = config.OPENAI_API_KEY_PATH # Use path from config


def process_articles():
    """ Takes content through a series of prompts that: deduplicate stories, determine their relevance to product management, and generate summaries for relevant stories. Also generates an intro
    for the email using the combined summaries and a daily prompt
    :return: dict
    """
    # Load the embedding model
    embedding_model = load_embedding_model()
    if not embedding_model:
        print("Failed to load embedding model. Uniqueness check and embedding generation will be skipped.")

    # Initialize ChromaDB client and collection
    article_collection = None
    if embedding_model: # Only init DB if model is available, as it's needed for queries
        cleaned_topic_name_for_collection = re.sub(r'\W+', '_', config.TARGET_TOPIC_NAME.lower())
        chroma_client = init_chroma_client(db_path=config.VECTOR_DB_PATH)
        if chroma_client:
            collection_name = f"{cleaned_topic_name_for_collection}_articles"
            article_collection = get_or_create_collection(client=chroma_client, collection_name=collection_name)
            if not article_collection:
                print(f"Failed to get or create ChromaDB collection '{collection_name}'. Uniqueness check will be skipped.")
        else:
            print("Failed to initialize ChromaDB client. Uniqueness check will be skipped.")

    scraped_articles = scrape_articles()


    def filter_articles_topics(article_text):
        """ This sends the article text to GPT where it evaluates if the topic should be included. Relevant articles lead to a response of 'relevant', irrelevant articles lead to a
        response of 'irrelevant'. The articles with the 'irrelevant' tag are filtered out of the set of articles to include in the email.
        :param str article_text: This is the text we want to evaluate in terms of relevant topics
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 1500  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with filtering articles
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": f"Analyze the text provided to determine if the article should be included in our content. We are screening out anything in the following categories: {', '.join(config.FORBIDDEN_TOPICS)}. If the primary focus is one of these, respond with: 'irrelevant'. If not, respond with 'relevant'. Only respond with 'relevant' or 'irrelevant'."},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=300,
                    temperature=.8,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                print(summary)
                return summary

            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None
    def filter_articles_relevance(article_text):
        """ This sends the article text to GPT where it evaluates it for relevance to product management. Relevant articles lead to a response of 'product related', irrelevant artielcles lead to a
        response of 'not product related'. The articles with the 'not product related' tag are filtered out of the set of articles to include in the email.
        :param str article_text: This is the text we want to evaluate in terms of relevance to product management
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 1500  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with filtering articles
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": f"Analyze the text provided to determine its relevance to {config.TARGET_TOPIC_NAME}.\n\n"
                                    f"- Consider the main theme of the article. Ask yourself: Does the theme relate to any aspect of {config.TARGET_TOPIC_NAME}, including concepts like: {', '.join(config.TARGET_TOPIC_KEYWORDS)}? Additionally, consider any broader connections to technology or business that might be relevant to professionals interested in {config.TARGET_TOPIC_NAME}, respond with: '{config.RELEVANCE_LABEL_POSITIVE}'.\n"
                                    f"- If the text is about highly niche technical details without broader implications for {config.TARGET_TOPIC_NAME}, evaluate if the context still implies a connection to {config.TARGET_TOPIC_NAME} decision-making or responsibilities. If it does, respond with '{config.RELEVANCE_LABEL_POSITIVE}'; if not, '{config.RELEVANCE_LABEL_NEGATIVE}'.\n"
                                    f"- If none of the above conditions are met, and the text doesn't focus on {config.TARGET_TOPIC_NAME} or its related concepts, respond with: '{config.RELEVANCE_LABEL_NEGATIVE}'."},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=300,
                    temperature=.8,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                return summary

            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None

    def remove_duplicates(article_set):
        """ Many sources may cover the same story in a given day. When this happens we only want to include the story one time using one source. The URLs for all scraped articles are sent to GPT
        and using the keywords in the URL it determines which URLs represent unique stories. In cases where there are multiple sources for a story it selects what it believes to be just the most
        reputable source and returns that URL.
        :param str article_set: All the URLs for the scraped articles are combined into a string so they can be sent to GPT for evaluation
        :return: list
        """
        for attempt in range(5):
            try:
                # In this case I used gpt-3.5-turbo-16k because I needed to send all URLs at once so the needed token count was much higher. But this model is twice as expensive so where possible I
                # used gpt-3.5-turbo
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system",
                         "content": "You will receive a set of URLs. Analyze the keywords within each URL to identify potential overlapping content. If multiple URLs seem to discuss the same topic based on shared keywords (for instance, if 3 URLs contain the terms 'microsoft' and 'teams'), choose only one URL, giving preference to the most reputable source based on general knowledge about the source's reputation. After your analysis, provide a comma-separated list of unique URLs that correspond to distinct topics. Your response should only be the list of URLs, without any additional text, line breaks, or '\n' characters."},
                        {"role": "user",
                         "content": article_set}
                    ],
                    max_tokens=10000,
                    temperature=.2,
                )
                deduped_urls = response["choices"][0]["message"]["content"].replace('\n', '').strip()  # GPT only returns things in string format. So though the prompt asks for a column
                # separated list, the list actually comes back as a string that you need to parse. On occasion GPT was appending a \n to each URL which caused the subsequent parsing and matching to
                # break. In the case that happens, this strips out the \n

                return deduped_urls
            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None
  
    def summary_generator(article_text):
        """ This sends the text of each article to GPT to have a summary generated
        :param str article_text: This is the text of an article that has been deemed relevant to product management
        :return: str
        """
        # Truncate the text to a certain length if necessary. This avoids sending content over the allowed token count and helps control cost
        max_length = 2000  # Adjust this value as needed
        if len(article_text) > max_length:
            article_text = article_text[:max_length]
            print(f"Article text too long. Truncated to {max_length} characters.")
        # Proceed with generating the summary
        for attempt in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": f"As an AI specializing in text analysis with a focus on {config.TARGET_TOPIC_NAME}, your job is to summarize the provided text. You should generate a one sentence summary for the text. This summary should first outline the topic of the article and then describe why this article is relevant to professionals interested in {config.TARGET_TOPIC_NAME}. The summaries should have a casual and fun but informed tone."},
                        {"role": "user",
                         "content": article_text}
                    ],
                    max_tokens=200,
                    temperature=.7,
                )
                summary = response["choices"][0]["message"]["content"].strip()
                return summary
            except openai.error.ServiceUnavailableError:
                wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                print(f"Server error, retrying in {wait_time} seconds")
                time.sleep(wait_time)
        print("Failed to generate response after several retries")
        return None

    def create_email_intro():
        """ The summaries for all the articles to be included in the email are passed into here and combined with a theme unique to the day of the week. These are sent to GPT where it creates a
        intro for the email.
        :return: str
        """
        current_date = datetime.now()
        day_of_week = current_date.weekday()
        # Unique theme for each day of the week
        if day_of_week == 0:
            theme = 'Encourage your readers to start the week with a positive, energized mindset. Lets get after it this week'
        elif day_of_week == 1:
            theme = 'Discuss unexpected turns and surprises in the field of product management.'
        elif day_of_week == 2:
            theme = 'Get through hump day with some witty banter and clever insights. Crack open an energy drink and power up to get through the week'
        elif day_of_week == 3:
            theme = 'Stimulate your neurons with some brain-teasing content. Maybe crack open an energy drink'
        elif day_of_week == 4:
            theme = 'Explore future technologies that could disrupt the field of product management.'
        elif day_of_week == 5:
            theme = 'Ride the wave of knowledge and insights from the week. Do something daring'
        else:
            theme = 'Use today to rest so you can be more productive the rest of the week. But be sure to stay up to date with what is happening by reading the newsletter. Sip some tea ' \
                    'while you relax'
        system_prompt = f"Hey there, AI! You're helping out with an intro for a daily {config.TARGET_TOPIC_NAME} newsletter called '{config.NEWSLETTER_NAME}'. Today's theme is '{theme}'—pretty cool, " \
                        f"right? Don't mention the theme name " \
                        f"directly but instead incorporate it into the vibe of the overall intro. You'll get to chew on different " \
                        f"topics tied to the theme. Your job is to whip up an engaging and fun intro that'll give the readers a taste of what's to come in the newsletter. Don't get too caught up in the details of each article—just give a general vibe of the content ahead. And hey, feel free to drop in a casual joke or use emojis when it fits. We're all about keeping our readers alert and on their toes! Remember, the goal is to make professionals interested in {config.TARGET_TOPIC_NAME} feel like they're kicking back with a can of knowledge that'll help them crush it in their work week. Let's make this exciting! Limit the intro to two sentences in length"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", # TODO: Consider making model names configurable in newsletter_config.py
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": intro_text}
            ],
            max_tokens=200,
            temperature=.7,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary

    # Create a list of unique URLs (initial deduplication based on URL)
    url_set = [f'{scraped_articles[key]["url"]},' for key in scraped_articles if scraped_articles[key].get("url")]
    if url_set:
        deduped_article_urls = [url.strip() for url in remove_duplicates(str(url_set)).split(',') if url.strip()]

        keys_to_delete_initial_dedup = []
        for key in list(scraped_articles.keys()):
            if scraped_articles[key].get("url") not in deduped_article_urls:
                keys_to_delete_initial_dedup.append(key)

        for key in keys_to_delete_initial_dedup:
            if key in scraped_articles: # Check if key still exists
                 del scraped_articles[key]
    else:
        print("No URLs found in scraped_articles for initial deduplication.")


    # Semantic Uniqueness Check using ChromaDB (before expensive LLM calls for topic/relevance)
    if article_collection and embedding_model:
        print("\nPerforming semantic uniqueness check against ChromaDB...")
        keys_to_delete_due_to_repeat = []
        for article_key, article_data in list(scraped_articles.items()): # Iterate over a copy for safe deletion
            current_article_text = article_data.get('text')
            if not current_article_text:
                print(f"Article {article_key} has no text, skipping uniqueness query.")
                continue

            # Generate a temporary embedding for the current article's text (truncated)
            query_text = current_article_text[:config.MAX_TEXT_FOR_EMBEDDING_QUERY]
            candidate_embedding_array = get_embedding(query_text, embedding_model)

            if candidate_embedding_array is None:
                print(f"Could not generate query embedding for {article_key}, skipping uniqueness check for this item.")
                continue

            candidate_embedding_list = candidate_embedding_array.tolist() if hasattr(candidate_embedding_array, 'tolist') else candidate_embedding_array

            try:
                similar_results = query_similar_articles(
                    collection=article_collection,
                    query_embedding=candidate_embedding_list,
                    n_results=3 # Fetch a few results to check against
                )
            except Exception as e:
                print(f"Error querying ChromaDB for article {article_key}: {e}. Skipping uniqueness for this item.")
                continue

            is_repeat = False
            if similar_results and similar_results.get('ids') and similar_results['ids'][0]:
                for i, past_article_id in enumerate(similar_results['ids'][0]):
                    distance = similar_results['distances'][0][i] if similar_results.get('distances') and similar_results['distances'][0] else None
                    metadata = similar_results['metadatas'][0][i] if similar_results.get('metadatas') and similar_results['metadatas'][0] else {}

                    publish_date_str = metadata.get('publish_date')

                    if publish_date_str and distance is not None:
                        try:
                            past_publish_date = datetime.date.fromisoformat(publish_date_str)
                            days_diff = (datetime.date.today() - past_publish_date).days

                            # Standard Recency and Similarity Check for potential repeat
                            if days_diff <= config.RECENCY_DAYS_FOR_UNIQUENESS and distance < config.SIMILARITY_THRESHOLD:
                                print(f"Potential repeat: Article '{article_data.get('title', article_key)}' is similar to '{metadata.get('title', past_article_id)}' (Published {days_diff} days ago, Distance: {distance:.4f}).")

                                # Now, check if it's a continuation or a rehash
                                if config.ENABLE_CONTINUATION_CHECK and days_diff <= config.MAX_DAYS_FOR_CONTINUATION_CHECK:
                                    old_summary = metadata.get('summary')
                                    if not old_summary:
                                        print(f"  Old summary for {past_article_id} not found in ChromaDB metadata. Cannot assess continuation. Treating as rehash for safety.")
                                        is_repeat = True # Treat as standard repeat
                                    else:
                                        print(f"  Assessing for continuation against old summary of {past_article_id}...")
                                        decision, justification = assess_topic_development(
                                            old_article_summary=old_summary,
                                            new_article_text=current_article_text,
                                            target_topic=config.TARGET_TOPIC_NAME
                                        )
                                        print(f"  Continuation Assessment for '{article_data.get('title', article_key)}': {decision} - {justification}")
                                        if decision == 'SIGNIFICANT DEVELOPMENT':
                                            is_repeat = False
                                            print(f"  Outcome: Marked as SIGNIFICANT DEVELOPMENT. Will keep '{article_data.get('title', article_key)}'.")
                                        else:
                                            is_repeat = True
                                            print(f"  Outcome: Marked as REHASH or assessment error. Will discard '{article_data.get('title', article_key)}'.")
                                else:
                                    is_repeat = True
                                    print(f"  Outcome: Standard repeat (not eligible for continuation check or check disabled). Will discard '{article_data.get('title', article_key)}'.")

                                if is_repeat:
                                    break
                            # No need for 'else: is_repeat = False' here.

                        except ValueError:
                            print(f"Could not parse publish_date '{publish_date_str}' for past article {past_article_id}. Skipping this specific comparison.")

            if is_repeat: # If the article was flagged as a repeat after all checks
                keys_to_delete_due_to_repeat.append(article_key)

        if keys_to_delete_due_to_repeat:
            print(f"\nRemoving {len(keys_to_delete_due_to_repeat)} articles due to being semantic repeats (after continuation check):")
            for key_to_delete in keys_to_delete_due_to_repeat:
                title_to_delete = scraped_articles.get(key_to_delete, {}).get('title', key_to_delete)
                print(f" - Deleting: {title_to_delete}")
                if key_to_delete in scraped_articles:
                    del scraped_articles[key_to_delete]
        else:
            print("No semantic repeats found in ChromaDB within the defined threshold and recency.")
    else:
        print("\nSkipping semantic uniqueness check: Embedding model or ChromaDB collection not available.")


    # LLM-based Filtering (Topics and Relevance)
    keys_to_delete_llm_filter = []
    for key, article_data in list(scraped_articles.items()): # Iterate over a copy
        article_text = article_data.get('text')
        if not article_text:
            print(f"Article {key} has no text, skipping LLM topic/relevance checks.")
            continue # Or add to keys_to_delete_llm_filter if articles without text are unwanted

        # LLM-based Topic Filtering (uses config.FORBIDDEN_TOPICS)
        topic_assessment_result = filter_articles_topics(article_text)
        if topic_assessment_result and 'irrelevant' in topic_assessment_result.lower():
            print(f"Article '{article_data.get('title', key)}' filtered out by topic check as '{topic_assessment_result}'.")
            keys_to_delete_llm_filter.append(key)
            continue

        # LLM-based Relevance Filtering (uses config.TARGET_TOPIC_NAME, config.TARGET_TOPIC_KEYWORDS,
        # config.RELEVANCE_LABEL_POSITIVE, config.RELEVANCE_LABEL_NEGATIVE)
        relevance_assessment_result = filter_articles_relevance(article_text)
        if relevance_assessment_result and config.RELEVANCE_LABEL_NEGATIVE in relevance_assessment_result.lower():
            print(f"Article '{article_data.get('title', key)}' filtered out by relevance check as '{relevance_assessment_result}'.")
            keys_to_delete_llm_filter.append(key)

    if keys_to_delete_llm_filter:
        print(f"\nRemoving {len(keys_to_delete_llm_filter)} articles after LLM topic/relevance filtering:")
        for key_to_delete in keys_to_delete_llm_filter:
            if key_to_delete in scraped_articles: # Check if key still exists
                 del scraped_articles[key_to_delete]


    # Generate summaries and final embeddings for relevant articles
    for key in list(scraped_articles.keys()):
        article_text = scraped_articles[key].get('text')
        if not article_text:
            # This case should ideally be caught earlier, but as a safeguard:
            print(f"Article {key} has no text at final processing stage. Removing.")
            del scraped_articles[key]
            continue

        summary = summary_generator(article_text) # LLM call for summary
        scraped_articles[key]['summary'] = summary

        # Generate and store final embedding (of the summary) for potential storage later
        if summary and embedding_model:
            final_embedding_array = get_embedding(summary, embedding_model)
            if final_embedding_array is not None:
                scraped_articles[key]['embedding'] = final_embedding_array.tolist() if hasattr(final_embedding_array, 'tolist') else final_embedding_array
            else:
                scraped_articles[key]['embedding'] = None
                print(f"Could not generate final embedding for article {key}'s summary.")
        elif not embedding_model:
            scraped_articles[key]['embedding'] = None
            print("Embedding model not available, skipping final embedding generation.")
        else: # Summary is None
            scraped_articles[key]['embedding'] = None
            print(f"Summary for article {key} was None, skipping final embedding generation.")


    print(scraped_articles)
    print('finished')
    intro_text = []
    for key_final in scraped_articles: # Iterate over the fully filtered and processed dict
        # Ensure summary exists before appending for intro text
        if scraped_articles[key_final].get('summary'): # Check if 'summary' key exists and is not None/empty
            intro_text.append(scraped_articles[key_final]['summary'])

    intro_text = str(intro_text)

    # Generate intro for the email
    email_intro = create_email_intro()

    return scraped_articles, email_intro


if __name__ == "__main__":
    articles, intro = process_articles()
    # Print results, ensuring keys exist
    for key, value in articles.items():
        title = value.get('title', 'N/A')
        url = value.get('url', 'N/A')
        summary = value.get('summary', 'N/A')
        embedding_info = "Present" if value.get('embedding') else "Absent"
        print(f"Title: {title} ({url})")
        print(f"Summary: {summary}")
        print(f"Embedding: {embedding_info}")
        print("-" * 30)
    print(f"\nEmail Intro:\n{intro}")


# --- Definition for Topic Development Assessment ---
# TODO: Consider moving to a dedicated llm_utils.py if more LLM helper functions are created.

def assess_topic_development(old_article_summary, new_article_text, target_topic, model="gpt-3.5-turbo", max_new_text_chars=4000):
    """
    Assesses if a new article represents a significant development or a rehash of an old article summary.

    Args:
        old_article_summary (str): Summary of the previously published article.
        new_article_text (str): Text of the new candidate article (can be truncated).
        target_topic (str): The overall newsletter topic for context.
        model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo".
        max_new_text_chars (int, optional): Maximum characters of the new article text to use.

    Returns:
        tuple: (str, str) containing the decision ('SIGNIFICANT DEVELOPMENT' or 'REHASH')
               and the justification. Returns (None, None) on error or if inputs are invalid.
    """
    if not all([old_article_summary, new_article_text, target_topic]):
        print("Error: Missing one or more required arguments for topic development assessment.")
        return None, None

    # Ensure openai.api_key or openai.api_key_path is configured
    # This check is conceptual; actual key setup is external to this function.
    if not openai.api_key and not openai.api_key_path:
        print("OpenAI API key not configured. Cannot assess topic development.")
        return "REHASH", "Skipping assessment due to missing OpenAI API key configuration." # Default to avoid error propagation

    truncated_new_article_text = new_article_text[:max_new_text_chars]

    prompt_content = f"""
Role: You are an expert news analyst determining if a new article provides a genuinely new update or is just a rehash of a previous story, specifically within the context of '{target_topic}'.

Context:
I am curating content for a newsletter about '{target_topic}'. I want to avoid sending my readers articles that are essentially repeats or minor updates of stories they've already seen recently.
You will be given the summary of an OLDER, already published article and the text of a NEWER, candidate article. Both articles are related to '{target_topic}'.

Input:
1. OLDER Article Summary: "{old_article_summary}"
2. NEWER Article Text (potentially truncated): "{truncated_new_article_text}"

Task:
Compare the NEWER Article Text against the OLDER Article Summary. Determine if the NEWER article introduces significant new information, developments, outcomes, or analyses that were not present in the OLDER article's summary.

- A 'SIGNIFICANT DEVELOPMENT' means the newer article provides substantial new facts, events, perspectives, data, or in-depth analysis that clearly moves the story forward. It's not just more details of the same core information, but a meaningful evolution of the topic.
- A 'REHASH' means the newer article largely repeats the information already covered in the older summary, perhaps with slightly different wording, more examples, or minor details that don't change the core understanding or outcome of the previous story.

Output Specification:
Your response MUST be in two parts, separated by "||".
1. Decision: Either "SIGNIFICANT DEVELOPMENT" or "REHASH".
2. Justification: A brief explanation (1-2 sentences) for your decision, highlighting what (if anything) is new or why it's considered a rehash.

Example Response:
REHASH||The new article discusses the same product launch mentioned in the old summary without adding new features or market reactions.
"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert news analyst determining content evolution."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=150,
            temperature=0.2, # Low temperature for more factual, less "creative" assessment
        )

        content = response["choices"][0]["message"]["content"].strip()

        parts = content.split("||", 1)
        if len(parts) == 2:
            decision = parts[0].strip().upper()
            justification = parts[1].strip()
            if decision in ["SIGNIFICANT DEVELOPMENT", "REHASH"]:
                return decision, justification
            else:
                print(f"Warning: LLM returned an unexpected decision type: '{decision}'. Content: {content}")
                return "REHASH", f"LLM returned unclassified decision: {decision}. Justification: {justification}"
        else:
            print(f"Warning: LLM response for topic development was not in the expected format: {content}")
            return "REHASH", f"LLM response format error. Original content: {content}"

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error during topic development assessment: {e}")
        # Defaulting to REHASH on API error to be conservative, could also be None
        return "REHASH", f"OpenAI API error: {str(e)}"
    except Exception as e:
        print(f"An unexpected error occurred during topic development assessment: {e}")
        return "REHASH", f"Unexpected error: {str(e)}"

# Conceptual test block for assess_topic_development, if run directly and OpenAI key is available
if __name__ == "__main__":
    # Existing main block for process_articles
    print("--- Running process_articles (main test block) ---")
    articles_processed, email_intro_processed = process_articles()
    for article_key, article_value in articles_processed.items():
        title = article_value.get('title', 'N/A')
        url = article_value.get('url', 'N/A')
        summary = article_value.get('summary', 'N/A')
        embedding_info = "Present" if article_value.get('embedding') else "Absent"
        print(f"Title: {title} ({url})")
        print(f"Summary: {summary}")
        print(f"Embedding: {embedding_info}")
        print("-" * 30)
    print(f"\nEmail Intro:\n{email_intro_processed}")

    # Separate, conceptual test for assess_topic_development
    # This part will only execute meaningfully if OpenAI key is configured.
    print("\n\n--- Conceptual Test for assess_topic_development ---")
    if not openai.api_key and not openai.api_key_path:
         print("OpenAI API key not configured. Skipping live API call test for assess_topic_development.")
    else:
        print("OpenAI API key found/path set. Running conceptual live test for assess_topic_development...")
        test_old_summary = "Company X launched Product Y, an AI-powered toothbrush, last week to mixed reviews."
        test_new_text_rehash = "Company X's new AI toothbrush, Product Y, is now available for purchase. It aims to improve dental hygiene using artificial intelligence."
        test_new_text_dev = "Following last week's launch, Company X today announced that its AI-powered toothbrush, Product Y, has received FDA approval for classifying plaque buildup, a feature previously undisclosed."
        test_target_topic = "AI Healthcare Gadgets"

        decision_rehash, just_rehash = assess_topic_development(test_old_summary, test_new_text_rehash, test_target_topic)
        print(f"\nTest Case REHASH (Expected: REHASH):")
        print(f"Decision: {decision_rehash}, Justification: {just_rehash}")

        decision_dev, just_dev = assess_topic_development(test_old_summary, test_new_text_dev, test_target_topic)
        print(f"\nTest Case SIGNIFICANT DEVELOPMENT (Expected: SIGNIFICANT DEVELOPMENT):")
        print(f"Decision: {decision_dev}, Justification: {just_dev}")



