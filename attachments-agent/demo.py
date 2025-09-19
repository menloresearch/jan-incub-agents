#!/usr/bin/env python3
"""
Demo script using the provided PDF document.
Question: 'How many data centers does the China's largest cloud provider have?'
"""

import os
from pathlib import Path
from processors import PyPDF2Processor
from document_agents import SimpleRAGDocumentAgent


def run_demo():
    """Run demo with the provided PDF document."""

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key: export OPENAI_API_KEY='your-api-key'")
        return

    print("=== SimpleRAGDocumentAgent Demo ===")
    print("Document: Campaign_038_Introducing_AC_Whitepaper_v5e.pdf")
    print(
        "Question: How many data centers does the China's largest cloud provider have?"
    )
    print()

    # Create the RAG agent
    agent = SimpleRAGDocumentAgent(
        model_id="gpt-4",
        base_url_llm=os.getenv("OPENAI_BASE_URL"),
        processor_class=PyPDF2Processor,
        processor_kwargs={},
        system_prompt="""You are an expert analyst specializing in cloud computing and data center infrastructure. 
        Your task is to carefully analyze documents and provide accurate, specific answers based on the content. 
        When answering questions about data centers or cloud providers, focus on extracting precise numerical data 
        and company-specific information. If you find relevant information, cite it clearly. If the information 
        is not available in the document, state that explicitly.""",
        # RAG Configuration
        chunk_size=400,  # Smaller chunks for more precise retrieval
        retrieval_method="bm25",
        retrieval_kwargs={"k1": 1.2, "b": 0.75},
        reranker_method=None,  # Start without reranking for simplicity
        # LLM parameters
        sampling_params={
            "temperature": 0.1,  # Low temperature for factual accuracy
            "max_tokens": 300,
        },
    )

    # Document path
    document_path = Path("data/Campaign_038_Introducing_AC_Whitepaper_v5e.pdf")

    if not document_path.exists():
        print(f"Error: Document not found at {document_path}")
        print("Please ensure the PDF file is in the data/ directory.")
        return

    # Question about China's largest cloud provider data centers
    chat_history = [
        {
            "role": "user",
            # "content": "How many data centers does the China's largest cloud provider have?",
            "content": "Which cities does the China's largest cloud provider have teams? Write the answer in list format with alphabetical rder.",
        }
    ]

    print("Processing document and generating response...")
    print("This may take a moment for first-time embedding model downloads...")
    print()

    try:
        response = agent.chat_with_document(
            chat_history=chat_history, document_file=document_path
        )

        print("=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)

    except Exception as e:
        print(f"Error during processing: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    # Run basic demo
    run_demo()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("To run your own questions, modify the chat_history in the script.")
    print("=" * 60)
