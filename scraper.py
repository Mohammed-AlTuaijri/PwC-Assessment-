import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

# Set to keep track of visited URLs to avoid duplicate scraping
visited = set()

# Check if the text can be encoded to UTF-8
def is_valid_text(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

# Remove any non-printable characters and filter non-UTF-8 characters
def clean_text(text):
    return ''.join(filter(lambda x: x.isprintable(), filter_non_utf8(text)))

# Filter out non-UTF-8 characters from the text
def filter_non_utf8(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

# Normalize the URL by parsing it and reconstructing it
def normalize_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path

# Check if the URL is valid by comparing it with the base URL and path prefix
def is_valid_link(url, base_url, path_prefix):
    normalized_url = normalize_url(url)
    parsed_url = urlparse(normalized_url)
    if parsed_url.netloc != urlparse(base_url).netloc:
        return False
    if not parsed_url.path.startswith(path_prefix):
        return False
    return bool(parsed_url.scheme)

# Recursively scrape the website starting from the given URL, within the specified depth
def scrape_website(url, base_url, path_prefix, depth):
    normalized_url = normalize_url(url)
    logging.debug(f"Scraping URL: {normalized_url} at depth {depth}")
    if normalized_url in visited or depth <= 0:
        logging.debug(f"Skipping already visited or depth-restricted URL: {normalized_url}")
        return ""

    visited.add(normalized_url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return "Connection error."
    
    soup = BeautifulSoup(response.content, 'html.parser')
    text_content = []
    for paragraph in soup.find_all('p'):
        text = paragraph.get_text(separator=" ", strip=True)
        if is_valid_text(text):
            cleaned_text = clean_text(text)
            text_content.append(cleaned_text)
    
    text_data = "\n".join(text_content)
    logging.debug(f"Extracted text length: {len(text_data)}")

    # Recursively scrape links
    link_count = 0
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        if is_valid_link(full_url, base_url, path_prefix):
            link_count += 1
            text_data += scrape_website(full_url, base_url, path_prefix, depth - 1)
            logging.debug(f"Concatenated text length: {len(text_data)}")
    
    logging.debug(f"Total links found and followed from {url}: {link_count}")
    return text_data