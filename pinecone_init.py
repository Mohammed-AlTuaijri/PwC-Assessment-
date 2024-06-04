import os
from pinecone import Pinecone, ServerlessSpec

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")


index_name = 'pinedb'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
index_info = pc.describe_index(index_name)
print(index_info)