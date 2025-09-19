#!/usr/bin/env python3
"""
Load MMLongBench-Doc dataset, filter for text-only examples, and prepare data directory.
"""

from datasets import load_dataset
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import snapshot_download

def ensure_data_directory():
    """Ensure data directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def load_and_filter_dataset() -> List[Dict[str, Any]]:
    """Load dataset and filter for text-only examples."""
    dataset = load_dataset("yubo2333/MMLongBench-Doc")['train']
    
    text_only_examples = []
    all_doc_ids = set()
    
    for example in dataset:
        all_doc_ids.add(example['doc_id'])
        
        try:
            # Parse evidence_sources (it's stored as a string representation of a list)
            evidence_str = example['evidence_sources']
            evidence_sources = eval(evidence_str) if evidence_str else []
        except:
            evidence_sources = []
        
        # Check if it's pure text
        is_text_only = True
        for source in evidence_sources:
            if 'Pure-text' not in str(source):
                is_text_only = False
                break
        
        # Additional check in question text for visual references
        question = example.get('question', '').lower()
        visual_words = ['chart', 'figure', 'image', 'graph', 'diagram', 'table', 'map', 'icon', 'visual', 'picture']
        if any(word in question for word in visual_words):
            is_text_only = False
        
        # Only include examples that have evidence and are text-only
        if is_text_only and evidence_sources:
            # Convert the example to a clean format
            clean_example = {
                'doc_id': example['doc_id'],
                'doc_type': example['doc_type'],
                'question': example['question'],
                'answer': example['answer'],
                'evidence_pages': eval(example['evidence_pages']) if example['evidence_pages'] else [],
                'evidence_sources': evidence_sources,
                'answer_format': example['answer_format']
            }
            text_only_examples.append(clean_example)
    
    return text_only_examples, sorted(all_doc_ids)

def check_available_documents(doc_ids: List[str], data_dir: Path) -> Dict[str, bool]:
    """Check which documents are already available in the data directory."""
    availability = {}
    for doc_id in doc_ids:
        doc_path = data_dir / doc_id
        availability[doc_id] = doc_path.exists()
    
    return availability

def download_all_documents(data_dir: Path) -> bool:
    """Download all documents from the Hugging Face repository."""
    try:
        print(f"\n🔄 Downloading all documents from Hugging Face repository...")
        print("This will download the entire documents folder...")
        
        # Download the documents folder from the repository
        repo_path = snapshot_download(
            repo_id="yubo2333/MMLongBench-Doc",
            repo_type="dataset",
            allow_patterns="documents/*",
            local_dir=".",
            force_download=True,
            resume_download=True
        )
        
        # Move documents from downloaded structure to our data directory
        downloaded_docs_dir = Path(repo_path) / "documents" 
        if not downloaded_docs_dir.exists():
            downloaded_docs_dir = Path("documents")
        
        if downloaded_docs_dir.exists():
            print(f"Moving documents from {downloaded_docs_dir} to {data_dir}")
            
            # Ensure data directory exists
            data_dir.mkdir(exist_ok=True)
            
            # Move all PDF files
            pdf_count = 0
            for pdf_file in downloaded_docs_dir.glob("*.pdf"):
                destination = data_dir / pdf_file.name
                if not destination.exists():
                    pdf_file.rename(destination)
                    pdf_count += 1
                else:
                    print(f"  Skipping {pdf_file.name} (already exists)")
            
            print(f"✅ Successfully moved {pdf_count} PDF documents to {data_dir}")
            
            # Clean up the downloaded folder structure
            try:
                if downloaded_docs_dir.name == "documents" and downloaded_docs_dir != data_dir:
                    import shutil
                    shutil.rmtree(downloaded_docs_dir)
            except:
                pass
            
            return True
        else:
            print(f"❌ Documents folder not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download documents: {e}")
        return False

def create_document_info(text_only_examples: List[Dict[str, Any]], all_doc_ids: List[str], data_dir: Path):
    """Create information about documents and their availability."""
    
    # Get unique doc IDs from text-only examples
    text_only_doc_ids = set(ex['doc_id'] for ex in text_only_examples)
    
    # Check availability
    availability = check_available_documents(all_doc_ids, data_dir)
    
    # Count examples per document
    doc_example_counts = {}
    doc_text_only_counts = {}
    
    # Load full dataset to get all examples per document
    full_dataset = load_dataset("yubo2333/MMLongBench-Doc")['train']
    for example in full_dataset:
        doc_id = example['doc_id']
        doc_example_counts[doc_id] = doc_example_counts.get(doc_id, 0) + 1
    
    # Count text-only examples per document
    for example in text_only_examples:
        doc_id = example['doc_id']
        doc_text_only_counts[doc_id] = doc_text_only_counts.get(doc_id, 0) + 1
    
    # Create document info
    document_info = {
        'summary': {
            'total_documents': len(all_doc_ids),
            'documents_with_text_only_examples': len(text_only_doc_ids),
            'documents_available_locally': sum(availability.values()),
            'total_examples': sum(doc_example_counts.values()),
            'text_only_examples': len(text_only_examples)
        },
        'documents': []
    }
    
    for doc_id in sorted(all_doc_ids):
        doc_info = {
            'doc_id': doc_id,
            'available_locally': availability[doc_id],
            'total_examples': doc_example_counts.get(doc_id, 0),
            'text_only_examples': doc_text_only_counts.get(doc_id, 0),
            'has_text_only': doc_id in text_only_doc_ids
        }
        document_info['documents'].append(doc_info)
    
    return document_info

def main():
    """Main function to load dataset and prepare data directory."""
    print("Loading MMLongBench-Doc dataset...")
    
    # Ensure data directory exists
    data_dir = ensure_data_directory()
    print(f"Data directory: {data_dir.absolute()}")
    
    # Load and filter dataset
    text_only_examples, all_doc_ids = load_and_filter_dataset()
    
    print(f"Found {len(text_only_examples)} text-only examples from {len(all_doc_ids)} total documents")
    
    # Check if we need to download documents
    availability = check_available_documents(all_doc_ids, data_dir)
    missing_docs = [doc_id for doc_id, available in availability.items() if not available]
    
    if missing_docs:
        print(f"\n📥 {len(missing_docs)} documents need to be downloaded")
        download_success = download_all_documents(data_dir)
        if not download_success:
            print("Warning: Document download failed, evaluation will be limited to available documents")
    else:
        print("\n✅ All documents already available locally")
    
    # Create document information after downloads
    doc_info = create_document_info(text_only_examples, all_doc_ids, data_dir)
    
    print(f"\nDocument availability summary:")
    print(f"  Total documents: {doc_info['summary']['total_documents']}")
    print(f"  Documents with text-only examples: {doc_info['summary']['documents_with_text_only_examples']}")
    print(f"  Documents available locally: {doc_info['summary']['documents_available_locally']}")
    print(f"  Total examples in dataset: {doc_info['summary']['total_examples']}")
    print(f"  Text-only examples: {doc_info['summary']['text_only_examples']}")
    
    # Show some examples
    print("\nSample text-only examples:")
    for i, example in enumerate(text_only_examples[:3]):
        doc_path = data_dir / example['doc_id']
        status = "✅ Available" if doc_path.exists() else "❌ Missing"
        
        print(f"\nExample {i+1}: {status}")
        print(f"  Doc ID: {example['doc_id']}")
        print(f"  Question: {example['question']}")
        print(f"  Answer: {example['answer']}")
    
    # Save outputs
    with open('text_only_eval_examples.json', 'w') as f:
        json.dump(text_only_examples, f, indent=2)
    
    with open('document_info.json', 'w') as f:
        json.dump(doc_info, f, indent=2)
    
    with open('all_document_ids.txt', 'w') as f:
        f.write('\n'.join(all_doc_ids))
    
    print(f"\nOutput files created:")
    print(f"  text_only_eval_examples.json - {len(text_only_examples)} filtered examples")
    print(f"  document_info.json - Document availability and statistics")
    print(f"  all_document_ids.txt - List of all {len(all_doc_ids)} document IDs")
    
    # Instructions for getting documents
    missing_docs = [doc_id for doc_id in all_doc_ids if not (data_dir / doc_id).exists()]
    if missing_docs:
        print(f"\n📋 To complete the evaluation setup:")
        print(f"  1. Obtain {len(missing_docs)} missing PDF documents")
        print(f"  2. Place them in the data/ directory")
        print(f"  3. Documents needed for text-only examples:")
        
        text_only_doc_ids = set(ex['doc_id'] for ex in text_only_examples)
        missing_text_only = [doc for doc in missing_docs if doc in text_only_doc_ids]
        
        if missing_text_only:
            print(f"     Priority (needed for text-only evaluation): {len(missing_text_only)} documents")
            for doc_id in sorted(missing_text_only)[:5]:
                print(f"       - {doc_id}")
            if len(missing_text_only) > 5:
                print(f"       ... and {len(missing_text_only) - 5} more")
    
    return text_only_examples

if __name__ == "__main__":
    main()