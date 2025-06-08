import datetime
import re # For cleaning collection name
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition, ContentId
from product_hunt_scraper import get_ph_grouped_products
from reddit_scraper import get_reddit_posts
from long_form_scraper import scrape_gmail_articles
from summary_generator import process_articles
from get_bcc_contacts import get_contact_emails_from_list_name
from send_grid_config import SENDGRID_API_KEY
import base64
import time
import pytz

# ChromaDB Integration
from vector_db_utils import init_chroma_client, get_or_create_collection, add_article_embedding
import newsletter_config as config


def time_until_next_run(target_hour, target_minute):
    """
    :param int target_hour: The hour of the day we want the email to be generated
    :param int target_minute: The minute of the hour we want the email to be generated
    :return: int
    """
    mountain = pytz.timezone('US/Mountain')  # Define the Mountain Time zone
    now = datetime.datetime.now(mountain)  # Get current time in Mountain Time
    target_time = datetime.datetime(now.year, now.month, now.day, target_hour, target_minute, tzinfo=mountain)

    if now > target_time:
        target_time = target_time + datetime.timedelta(days=1)

    return (target_time - now).total_seconds()


def run_newsletter_script():
    """ Gathers all the content, formats the email, and sends
    :return:
    """
    ### INITIALIZE ChromaDB ###
    # Clean TARGET_TOPIC_NAME from config for use as a collection name
    # Replace spaces and special characters with underscores, and lowercase
    cleaned_topic_name = re.sub(r'\W+', '_', config.TARGET_TOPIC_NAME.lower())

    chroma_client = init_chroma_client(db_path=config.VECTOR_DB_PATH)
    article_collection = None
    if chroma_client:
        collection_name = f"{cleaned_topic_name}_articles"
        article_collection = get_or_create_collection(client=chroma_client, collection_name=collection_name)
        if not article_collection:
            print(f"Failed to get or create ChromaDB collection '{collection_name}'. Embeddings will not be stored.")
    else:
        print("Failed to initialize ChromaDB client. Embeddings will not be stored.")

    ### GET CONTENT FOR EMAIL ###
    articles, intro = process_articles() # This should now include embeddings in 'articles'
    scraped_articles = articles
    intro_text = intro

    ph_items = {} # Initialize as empty dict
    if config.ENABLE_PRODUCT_HUNT_SCRAPER:
        ph_items = get_ph_grouped_products()
    else:
        print("Product Hunt scraper disabled via config.")

    reddit_posts = get_reddit_posts()
    long_reads = scrape_gmail_articles()
    image_path = "PATH TO THE IMAGE YOU WANT TO INCLUDE AT TOP OF EMAIL"
    list_name = 'NAME OF SENDGRID CONTACT LIST YOU WISH TO SEND TO'

    def format_html_email(intro_text, scraped_articles,ph_items, reddit_posts,
                          long_reads):
        """ Formats the contents of the email into the desired layout
        :param str intro_text: The intro for the email
        :param dict scraped_articles: The url, title, and summary for the Top Stores portion of the email
        :param dict ph_items: The url, title, and summary for the Products to Watch portion of the email
        :param dict reddit_posts: The url and title for the Product Discussion portion of the email
        :param dict long_reads: The url and title for the Long Reads portion of the email
        :return:
        """
        today_date = datetime.date.today().strftime('%B %d, %Y')  # Depending on when email is sent (late at night/early morning will dictate which of these date options to use)
        tomorrow_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%B %d, %Y')

        # Centered image with specified width
        html_content = '<img src="cid:image_cid" style="display:block; margin-left:auto; margin-right:auto; width:400px;">'

        # Centered date
        html_content += f'<p style="text-align:center;">{today_date}</p>'

        html_content += f'<p>{intro_text}</p>'

        # Section heading and daily news dictionary content
        html_content += '<h3>Top Stories</h3>'
        for key, value in scraped_articles.items():
            html_content += f'<a href="{value["url"]}">{value["title"]}</a>'
            html_content += f'<p>{value["summary"]}</p>'

        # Section heading and product hunt content
        html_content += '<h3>Products to Watch</h3>'
        for key, value in ph_items.items():
            html_content += f'<a href="{value["link"]}">{value["title"]}</a>: {value["desc"]}<br>'

        html_content += '<h3>Product Discussions</h3>'
        for key, value in reddit_posts.items():
            html_content += f'<a href="{value["url"]}">{value["title"]}</a><br>'

        if long_reads: # Long Reads will sometimes be empty, since and empty dict is falsey this will only be included there are items in the dict
            html_content += '<h3>Long Reads</h3>'
            for key, value in long_reads.items():
                html_content += f'<a href="{value["url"]}">{value["subject"]}</a><br>'

        html_content += '<hr>'
        html_content += '<p style="text-align:center;">What else would you like to see here? Send feedback to pmamnews@gmail.com</p>'
        # Subscribe link
        html_content += '<p style="text-align:center;">Did someone forward this to you? Sign up so that you never miss the next big thing! <a href="https://pmnews.today">Subscribe</a></p>'
        html_content += '<hr>'

        return html_content

    def send_email(recipient_emails, intro_text, scraped_articles, ph_items, reddit_posts,
                          long_reads, image_path, sendgrid_api_key):
        """ Send the email based on recipient list and html configuration
        :param list recipient_emails: The list of individuals you are sending the email to
        :param str intro_text: The intro for the email
        :param dict scraped_articles: The url, title, and summary for the Top Stores portion of the email
        :param dict ph_items: The url, title, and summary for the Products to Watch portion of the email
        :param dict reddit_posts: The url and title for the Product Discussion portion of the email
        :param dict long_reads: The url and title for the Long Reads portion of the email
        :param str image_path: The image to include at the top of the email
        :param str sendgrid_api_key: Your API key for SendGrid
        :return:
        """

        # Formatting HTML
        html_content = format_html_email(intro_text, scraped_articles, ph_items, reddit_posts, long_reads)

        # Create the Mail object
        message = Mail(
            from_email='pmamnews@gmail.com',  # TODO: Consider moving to config if it changes per topic
            to_emails='pmamnews@gmail.com',  # TODO: Consider moving to config if it changes per topic
            subject=f"{config.NEWSLETTER_NAME} - {datetime.date.today().strftime('%B %d, %Y')}",
            html_content=html_content
        )

        # Add each email as BCC
        for email in recipient_emails:
            message.add_bcc(email)

        # Attach the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
            f.close()

        encoded_image = base64.b64encode(image_data).decode()
        attached_image = Attachment(
            FileContent(encoded_image),
            FileName('image.jpg'),
            FileType('image/jpeg'),
            Disposition('inline'),
            ContentId('image_cid')
        )
        message.attachment = attached_image

        try:
            sg = SendGridAPIClient(sendgrid_api_key)
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print("Error sending the email:", e)
            if hasattr(e, 'body'):
                print(e.body)

    recipients = get_contact_emails_from_list_name(SENDGRID_API_KEY, list_name)

    # Store embeddings before sending the email
    if article_collection and scraped_articles:
        print("\nStoring article embeddings in ChromaDB...")
        today_date_str = datetime.date.today().isoformat()
        articles_stored_count = 0
        for article_url, article_data in scraped_articles.items():
            if article_data and article_data.get('embedding') and article_data.get('title') and article_data.get('summary'):
                # Ensure embedding is a list of floats, not None or other types
                embedding_to_store = article_data['embedding']
                if not isinstance(embedding_to_store, list):
                    # This can happen if embedding failed and was set to None, or if it's still a numpy array
                    print(f"Skipping {article_url} - embedding is not a list or is None.")
                    continue

                metadata = {
                    'publish_date': today_date_str,
                    'title': article_data['title'],
                    'url': article_url, # article_url is the key from scraped_articles
                    'summary': article_data['summary'], # Storing summary as well for context
                    'source_topic': config.TARGET_TOPIC_NAME
                }
                if add_article_embedding(collection=article_collection, embedding=embedding_to_store, metadata=metadata, article_id=article_url):
                    articles_stored_count +=1
                else:
                    print(f"Failed to store embedding for {article_url}.")
            else:
                print(f"Skipping storing embedding for {article_url} due to missing embedding, title, or summary.")
        print(f"Successfully stored embeddings for {articles_stored_count} articles.")
    elif not article_collection:
        print("ChromaDB collection not available. Skipping storing embeddings.")
    else:
        print("No articles to store embeddings for.")

    send_email(recipients, intro_text, scraped_articles, ph_items, reddit_posts, long_reads, image_path, SENDGRID_API_KEY)

### CRON TO CONTROL SEND OF EMAIL ###


while True:
    sleep_duration = time_until_next_run(12, 15)  # Set the time you want the email to generate at
    print(f"Sleeping for {sleep_duration / 60} minutes...")
    time.sleep(sleep_duration)

    run_newsletter_script()

    time.sleep(60)
