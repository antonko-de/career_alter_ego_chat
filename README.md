# Career Alter Ego Chat

An AI-powered chat interface that uses RAG (Retrieval Augmented Generation) to provide personalized responses based on professional background and experience.

## Features

- Semantic search using OpenAI embeddings
- Document chunking and processing
- FAISS vector store for efficient retrieval
- Gradio web interface
- Environment variable configuration

## Setup

1. Clone the repository
2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
PUSHOVER_TOKEN=your_pushover_token_here
PUSHOVER_USER=your_pushover_user_here
```

## Running Locally

```bash
uv run app.py
```

## Hugging Face Spaces

This app is configured to run on Hugging Face Spaces. The space will automatically:
- Load environment variables from Space settings
- Run the Gradio interface

## Environment Variables

The following environment variables need to be set in your Hugging Face Space settings:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PUSHOVER_TOKEN`: Your Pushover API token (optional)
- `PUSHOVER_USER`: Your Pushover user key (optional) 