Overview

This project is an AI Query Interface designed to interact with various language models (LLMs) to provide users with responses based on their queries. The primary functionality involves scraping text from websites, embedding the text into a vector database, and querying the database to find relevant information to respond to user queries. The responses are generated using different AI models, including GPT-3.5-Turbo, GPT-4, and llama-2-70b-chat.

Features:
  - Web Scraping: The application can scrape text data from a specified website.
  - Text Embedding: Scraped text is embedded into a vector database using Pinecone.
  - Query Processing: User queries are processed to retrieve relevant information from the vector database.
  - Model Integration: The application integrates with multiple language models to generate responses based on the user input and the retrieved information.
  - Real-time Streaming: Uses Server-Sent Events (SSE) to enable real-time streaming of responses to the frontend (only works with GPT-3.5-Turbo, GPT-4).

Challenges and Limitations

  - Falcon Model Integration: Initially intended to include the Falcon model "falcon-40b-instruct" for generating responses. However, due to time constraints, this feature was not fully implemented.
  - Evaluation Mechanism: Planned to implement a mechanism to compare and evaluate the outputs from different models to determine the best-performing LLM for the given user input. Unfortunately, this feature could not be added
    within the project timeline.
  - First-time Implementation: This is my first time working on a project of this nature. As a result, the application may have several issues and areas for improvement.
