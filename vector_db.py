from pinecone_init import index
import openai
import logging
from ratelimit import limits, sleep_and_retry
import os
import tiktoken

openai.api_key = os.getenv('OPENAI_API_KEY')
encoding = tiktoken.encoding_for_model("text-embedding-3-large")

# max_tokens for embedding to avoid exceeding context length
MAX_TOKENS_FOR_EMBEDDING = 8091

# Remove non-printable characters from text
def clean_text(text):
    return ''.join(filter(lambda x: x.isprintable(), text))

# Chunk text into smaller parts suitable for embedding
def chunk_text_for_embedding(text, max_tokens=MAX_TOKENS_FOR_EMBEDDING):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_tokens = encoding.encode(word)
        word_length = len(word_tokens)
        if current_length + word_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Define the rate limit for RPM
RATE_LIMIT = 3000
RATE_PERIOD = 60  # in seconds


# Embed a single chunk of text
@sleep_and_retry
@limits(calls=3000, period=60)
def embed_chunk(chunk):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=chunk
    )
    return response.data[0].embedding

# Embed the entire text by chunking it first
def embed_text(text):
    chunks = chunk_text_for_embedding(text)
    embeddings = []
    for chunk in chunks:
        if len(encoding.encode(chunk)) <= MAX_TOKENS_FOR_EMBEDDING:
            embeddings.append(embed_chunk(chunk))
        else:
            logging.error(f"Chunk exceeds the max token limit: {chunk}")
    return embeddings

# Store the embedded text in the Pinecone vector database
def store_in_vector_db(text):
    chunks = chunk_text_for_embedding(text)
    embeddings = embed_text(text)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if len(embedding) != 3072:  # embedding size
            logging.error(f"Embedding dimension mismatch: expected 3072, got {len(embedding)}")
            continue
        # Store the embedding in Pinecone along with metadata
        index.upsert([(str(i), embedding, {"text": chunk})])

# Query the vector database to retrieve similar text chunks
def query_vector_db(query):
    query_embedding = embed_chunk(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    results = [match['metadata']['text'] for match in response['matches']]
    logging.debug(f"Retrieved text chunks: {results}")
    return results