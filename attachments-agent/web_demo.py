#!/usr/bin/env python3
"""
Beautiful Streamlit Web Demo for Document Q&A System
Allows users to upload PDF documents, select configurations, and ask questions.
"""

import os
import streamlit as st
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import json
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

from processors import PyPDF2Processor
from document_agents import SimpleRAGDocumentAgent

# Configure Streamlit page
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid #ff6b6b;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: -1rem -1rem 2rem -1rem;
        padding: 2rem;
    }
    
    .config-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    .response-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f1f3f4;
        border-radius: 5px;
    }
    
    .upload-area {
        border: 2px dashed #4ecdc4;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .preview-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .preview-image {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    .document-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .page-selector {
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Load all available configuration files."""
    configs = {}
    config_dir = Path("config")

    if not config_dir.exists():
        st.error("Config directory not found!")
        return {}

    for config_file in config_dir.glob("*.yaml"):
        if config_file.name != "evaluation_suite.yaml":
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    # Use the name from config or filename as key
                    name = config.get("name", config_file.stem)
                    configs[name] = {"config": config, "filename": config_file.name}
            except Exception as e:
                st.warning(f"Could not load config {config_file.name}: {e}")

    return configs


def convert_pdf_to_images(pdf_bytes: bytes, max_pages: int = 5) -> List[Image.Image]:
    """Convert PDF pages to images for preview."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    # Limit to first few pages for performance
    num_pages = min(len(doc), max_pages)

    for page_num in range(num_pages):
        page = doc[page_num]
        # Render page to image with good quality
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    doc.close()
    return images


def get_pdf_info(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract PDF information."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    info = {
        "page_count": len(doc),
        "title": doc.metadata.get("title", "N/A"),
        "author": doc.metadata.get("author", "N/A"),
        "subject": doc.metadata.get("subject", "N/A"),
        "creator": doc.metadata.get("creator", "N/A"),
    }
    doc.close()
    return info


def create_agent_from_config(
    config: Dict[str, Any], model_id: str = "gpt-4"
) -> SimpleRAGDocumentAgent:
    """Create a SimpleRAGDocumentAgent from configuration."""
    return SimpleRAGDocumentAgent(
        model_id=model_id,
        base_url_llm=os.getenv("OPENAI_BASE_URL"),
        processor_class=PyPDF2Processor,
        processor_kwargs={},
        system_prompt="""You are an expert document analyst. Your task is to carefully analyze documents and provide accurate, specific answers based on the content. When answering questions, focus on extracting precise information and cite it clearly. If the information is not available in the document, state that explicitly.""",
        # RAG Configuration from config
        chunk_size=config.get("chunk_size", 400),
        retrieval_method=config.get("retrieval_method", "bm25"),
        retrieval_kwargs=config.get("retrieval_kwargs", {}),
        reranker_method=config.get("reranker_method"),
        reranker_kwargs=config.get("reranker_kwargs", {}),
        # LLM parameters
        sampling_params={
            "temperature": 0.1,
            "max_tokens": 500,
        },
    )


def main():
    # Header
    st.markdown(
        '<div class="main-header"><h1>📚 Document Q&A Assistant</h1><p>Upload a PDF, select a configuration, and ask questions about your document!</p></div>',
        unsafe_allow_html=True,
    )

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Check API key
        api_key_status = os.getenv("OPENAI_API_KEY") is not None
        if api_key_status:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not set")
            st.markdown(
                "Please set your API key: `export OPENAI_API_KEY='your-api-key'`"
            )
            return

        # Model selection
        model_id = st.selectbox("🤖 Select Model", ["gpt-4.1", "gpt-4.1-mini"], index=0)

        # Load configurations
        configs = load_all_configs()

        if not configs:
            st.error("No configurations found!")
            return

        # Configuration selection
        config_names = list(configs.keys())
        selected_config_name = st.selectbox(
            "📋 Select Configuration",
            config_names,
            help="Choose from 20+ different retrieval configurations",
        )

        selected_config = configs[selected_config_name]

        # Display selected configuration details
        with st.expander("📊 Configuration Details"):
            config_details = selected_config["config"]
            st.json(
                {
                    "name": config_details.get("name", "N/A"),
                    "chunk_size": config_details.get("chunk_size", "N/A"),
                    "retrieval_method": config_details.get("retrieval_method", "N/A"),
                    "reranker_method": config_details.get("reranker_method", "None"),
                    "filename": selected_config["filename"],
                }
            )

    # Main content area - Three column layout
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.markdown("## 📄 Upload Document")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type="pdf", help="Upload a PDF document to analyze"
        )

        if uploaded_file is not None:
            st.markdown(
                '<div class="success-message">✅ File uploaded successfully!</div>',
                unsafe_allow_html=True,
            )

            # Get PDF info
            pdf_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            pdf_info = get_pdf_info(pdf_bytes)

            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes",
                "Pages": pdf_info["page_count"],
                "Title": pdf_info["title"],
                "Author": pdf_info["author"],
            }

            with st.expander("📊 Document Details"):
                st.json(file_details)

    with col2:
        st.markdown("## 📖 Document Preview")

        if uploaded_file is not None:
            try:
                # Convert PDF to images for preview
                with st.spinner("🔄 Rendering document preview..."):
                    max_preview_pages = st.slider(
                        "Max pages to load",
                        1,
                        10,
                        3,
                        help="More pages = slower loading",
                    )
                    pdf_images = convert_pdf_to_images(
                        pdf_bytes, max_pages=max_preview_pages
                    )

                if pdf_images:
                    # Navigation controls
                    col_prev, col_page, col_next = st.columns([1, 2, 1])

                    with col_page:
                        if len(pdf_images) > 1:
                            selected_page = (
                                st.selectbox(
                                    "📄 Page",
                                    range(1, len(pdf_images) + 1),
                                    format_func=lambda x: f"Page {x} of {len(pdf_images)}",
                                )
                                - 1
                            )
                        else:
                            selected_page = 0
                            st.write(f"📄 Page 1 of {len(pdf_images)}")

                    # Navigation buttons
                    with col_prev:
                        if selected_page > 0:
                            if st.button("⬅️ Prev"):
                                st.rerun()

                    with col_next:
                        if selected_page < len(pdf_images) - 1:
                            if st.button("Next ➡️"):
                                st.rerun()

                    # Display selected page with container
                    st.markdown(
                        '<div class="preview-container">', unsafe_allow_html=True
                    )
                    st.image(
                        pdf_images[selected_page],
                        caption=f"📖 Page {selected_page + 1} - {uploaded_file.name}",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Document statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("📄 Total Pages", pdf_info["page_count"])
                    with col_stat2:
                        st.metric("👀 Previewing", len(pdf_images))

                    if len(pdf_images) < pdf_info["page_count"]:
                        st.info(
                            f"ℹ️ Showing first {len(pdf_images)} of {pdf_info['page_count']} pages. Increase slider above to load more."
                        )

            except Exception as e:
                st.error(f"❌ Could not preview document: {str(e)}")
                st.info(
                    "💡 Try uploading a different PDF file or check if the file is corrupted."
                )
        else:
            st.info("📄 Upload a PDF to see preview here")

    with col3:
        st.markdown("## 💬 Ask Questions")

        # Question input
        question = st.text_area(
            "Enter your question",
            placeholder="e.g., What is the main topic of this document?",
            height=120,
            help="Ask any question about the uploaded document",
        )

        # Process button
        process_button = st.button(
            "🚀 Ask Question", type="primary", use_container_width=True
        )

    # Processing and Results
    if uploaded_file is not None and question and process_button:
        if not api_key_status:
            st.error("Please set your OpenAI API key first!")
            return

        with st.spinner("🔍 Processing document and generating response..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = Path(tmp_file.name)

                # Create agent with selected configuration
                agent = create_agent_from_config(selected_config["config"], model_id)

                # Prepare chat history
                chat_history = [{"role": "user", "content": question}]

                # Get response
                response = agent.chat_with_document(
                    chat_history=chat_history, document_file=tmp_file_path
                )

                # Display results
                st.markdown("## 💡 Response")
                st.markdown(
                    f'<div class="response-card"><h3>🤖 AI Assistant Response</h3><p>{response}</p></div>',
                    unsafe_allow_html=True,
                )

                # Display configuration used
                with st.expander("🔧 Configuration Used"):
                    st.write(f"**Configuration:** {selected_config_name}")
                    st.write(f"**Model:** {model_id}")
                    st.write(f"**File:** {selected_config['filename']}")

                # Clean up temporary file
                os.unlink(tmp_file_path)

            except Exception as e:
                st.markdown(
                    f'<div class="error-message">❌ Error: {str(e)}</div>',
                    unsafe_allow_html=True,
                )
                st.error("Please check your configuration and try again.")

                # Clean up temporary file if it exists
                if "tmp_file_path" in locals() and tmp_file_path.exists():
                    os.unlink(tmp_file_path)

    # Footer with additional information
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
        <div style="text-align: center; color: #666;">
            <h4>🎯 How to Use</h4>
            <p><strong>1.</strong> Upload a PDF document<br>
            <strong>2.</strong> Preview your document with interactive viewer<br>
            <strong>3.</strong> Select a configuration from the sidebar<br>
            <strong>4.</strong> Ask a question about your document<br>
            <strong>5.</strong> Get AI-powered answers!</p>
            
            <h4 style="margin-top: 2rem;">📖 Preview Features</h4>
            <p>🔍 <strong>Interactive Navigation:</strong> Browse pages with prev/next buttons<br>
            📊 <strong>Page Selector:</strong> Jump to any page directly<br>
            ⚙️ <strong>Load Control:</strong> Adjust how many pages to preview<br>
            📄 <strong>Document Info:</strong> View metadata and page counts</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
