#!/usr/bin/env python3
"""
Evaluation pipeline for retrieval + reranking using MMLongBench-Doc dataset.
Uses GPT-4o-mini as LLM judge to compare retrieved chunks with expected answers.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import gc
import torch
import csv
import pandas as pd
import yaml
import time
from datetime import datetime
from tqdm import tqdm
from json_repair import repair_json
from document_agents import SimpleRAGDocumentAgent
from processors import PyPDF2Processor, MarkItDownProcessor, DoclingProcessor, NativeProcessor, MarkerProcessor
from utils import split_into_multi_chunks
from llm_wrapper import OpenAIApiWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

class RetrievalEvaluator:
    """Evaluator for retrieval + reranking systems."""

    def __init__(
        self,
        judge_model: str = "gpt-5-mini",
        base_url: Optional[str] = None,
        judge_workers: int = 4,
    ):
        self.model_id = "gpt-4o"  # just for tokenizer
        self.judge_model = judge_model
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.judge_workers = max(1, int(judge_workers))
        
        # Mapping of processor class names to actual classes
        self.processor_classes = {
            "PyPDF2Processor": PyPDF2Processor,
            "MarkItDownProcessor": MarkItDownProcessor,
            "DoclingProcessor": DoclingProcessor,
            "NativeProcessor": NativeProcessor,
            "MarkerProcessor": MarkerProcessor,
        }

        # Initialize LLM judge for chunk relevance evaluation
        self.judge_llm = OpenAIApiWrapper(
            model_id=self.judge_model,
            base_url=self.base_url,
            sampling_params={
                "temperature": 0.1,
                # "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
        )
        
        # Store current RAG agent for cleanup
        self._current_rag_agent = None

    def create_rag_agent(
        self,
        chunk_size: int = 400,
        retrieval_method: str = "bm25",
        retrieval_kwargs: Optional[Dict] = None,
        reranker_method: Optional[str] = None,
        reranker_kwargs: Optional[Dict] = None,
        processor_class: str = "PyPDF2Processor",
    ) -> SimpleRAGDocumentAgent:
        """Create a RAG agent with specified configuration."""
        
        # Convert string processor class name to actual class
        if isinstance(processor_class, str):
            actual_processor_class = self.processor_classes.get(processor_class)
            if actual_processor_class is None:
                raise ValueError(f"Unknown processor class: {processor_class}")
        else:
            actual_processor_class = processor_class

        return SimpleRAGDocumentAgent(
            model_id=self.model_id,
            base_url_llm=self.base_url,
            processor_class=actual_processor_class,
            processor_kwargs={},
            system_prompt="You are an expert document analyst.",
            chunk_size=chunk_size,
            retrieval_method=retrieval_method,
            retrieval_kwargs=retrieval_kwargs or {},
            reranker_method=reranker_method,
            reranker_kwargs=reranker_kwargs or {},
            sampling_params={"temperature": 0.1, "max_tokens": 300},
        )

    def judge_chunks_relevance_batch(
        self, question: str, chunks: List[str], expected_answer: str
    ) -> List[Dict[str, Any]]:
        """
        Use LLM judge to evaluate all retrieved chunks at once for relevance to answering the question.
        
        Args:
            question: The question being asked
            chunks: List of retrieved chunks to evaluate
            expected_answer: The expected answer for reference
        
        Returns:
            List of dicts with 'is_relevant' (bool) for each chunk in order
        """
        
        # Format chunks with indices for the prompt
        formatted_chunks = ""
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks += f"\n--- Chunk {i} ---\n{chunk}\n"
        
        judge_prompt = f"""
You are an expert judge evaluating the relevance of text chunks for answering questions.

Question: {question}
Expected Answer: {expected_answer}

Retrieved Text Chunks to Evaluate:
{formatted_chunks}

Task: Determine which chunks contain information that would help answer the question correctly.

Consider for each chunk:
1. Does the chunk contain facts, data, or context relevant to the question?
2. Would this chunk help generate the expected answer or something very similar?
3. Is the information in the chunk directly or indirectly related to what's being asked?

Important Notes:
- It's possible that NONE of the chunks are relevant (all should be marked false)
- It's possible that MULTIPLE chunks are relevant and need to be combined for a complete answer
- Each chunk should be evaluated independently

Respond with valid JSON in this exact format (just a simple array of true/false values):
{{
  "relevant": [true, false, true, false, false, true, false, false, false, false]
}}
"""

        try:
            messages = [{"role": "user", "content": judge_prompt}]
            response = self.judge_llm.generate(messages)

            # Parse JSON response using json-repair as default
            try:
                # Use json-repair by default for more robust parsing
                repaired_response = repair_json(response.strip())
                result = json.loads(repaired_response)
                
                # Extract relevance results from simplified format
                relevant_array = result.get("relevant", [])
                
                # Create results list in the same order as input chunks
                results = []
                for i in range(len(chunks)):
                    # Get relevance for this chunk (default to False if missing)
                    is_relevant = relevant_array[i] if i < len(relevant_array) else False
                    results.append({"is_relevant": bool(is_relevant)})
                
                return results

            except Exception as e:
                print(f"Failed to parse/repair JSON response: {e}")
                print(f"Raw response: {response}")

                # Fallback to default values for all chunks
                return [{"is_relevant": False} for _ in chunks]

        except Exception as e:
            print(f"Error in batch judge evaluation: {e}")
            return [{"is_relevant": False} for _ in chunks]
    
    def judge_chunk_relevance(
        self, question: str, chunk: str, expected_answer: str
    ) -> Dict[str, Any]:
        """
        Legacy method for single chunk evaluation - now uses batch method internally.
        Kept for backward compatibility.

        Returns:
            Dict with 'is_relevant' (bool)
        """
        results = self.judge_chunks_relevance_batch(question, [chunk], expected_answer)
        return results[0] if results else {"is_relevant": False}

    def evaluate_retrieval_for_example(
        self,
        example: Dict[str, Any],
        document_text: str,
        rag_agent: SimpleRAGDocumentAgent,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate retrieval performance for a single example.

        Returns evaluation metrics including MRR-related data and timing for core operations.
        """

        question = example["question"]
        expected_answer = example["answer"]

        # Start timing for core retrieval operations only
        core_start_time = time.time()

        # Split document into chunks (same as RAG agent)
        chunks = split_into_multi_chunks(document_text, chunk_size=rag_agent.chunk_size)

        if not chunks:
            core_time = time.time() - core_start_time
            return {
                "question": question,
                "expected_answer": expected_answer,
                "is_error": True,
                "error": "No chunks generated from document",
                "retrieved_chunks": [],
                "relevance_scores": [],
                "first_relevant_rank": None,
                "reciprocal_rank": 0.0,
                "map_data": {"relevant_ranks": [], "average_precision": 0.0},
            }, core_time

        # Retrieve chunks using the RAG agent's retrieval system
        retrieved_chunks = rag_agent.retrieval.retrieve(
            query=question, documents=chunks, top_k=10  # Retrieve more for evaluation
        )

        # Apply reranking if available
        if rag_agent.reranker:
            retrieved_chunks = rag_agent.reranker.rerank(
                query=question, documents=retrieved_chunks, top_k=10
            )

        # End timing for core operations (before judge calls)
        core_time = time.time() - core_start_time

        # Judge relevance of all retrieved chunks at once
        relevance_evaluations = []
        relevant_ranks = []

        # Use batch judging for all chunks at once
        try:
            batch_results = self.judge_chunks_relevance_batch(question, retrieved_chunks, expected_answer)
            
            # Process results and maintain rank order
            for rank, (chunk, judge_result) in enumerate(zip(retrieved_chunks, batch_results), 1):
                relevance_evaluations.append(
                    {
                        "rank": rank,
                        "chunk": chunk,
                        "is_relevant": judge_result["is_relevant"],
                    }
                )
                if judge_result["is_relevant"]:
                    relevant_ranks.append(rank)
                    
        except Exception as e:
            print(f"Error in batch judge evaluation: {e}")
            # Fallback to marking all as not relevant
            for rank, chunk in enumerate(retrieved_chunks, 1):
                relevance_evaluations.append(
                    {
                        "rank": rank,
                        "chunk": chunk,
                        "is_relevant": False,
                    }
                )

        # Calculate average precision (for MAP)
        average_precision = 0.0
        if relevant_ranks:
            precision_sum = 0.0
            for i, rank in enumerate(relevant_ranks):
                precision_at_rank = (i + 1) / rank
                precision_sum += precision_at_rank
            average_precision = precision_sum / len(relevant_ranks)

        # Calculate reciprocal rank (for MRR)
        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

        # Prepare detailed chunk information for JSONL logging
        detailed_chunks = []
        for evaluation in relevance_evaluations:
            detailed_chunks.append({
                "rank": evaluation["rank"],
                "chunk_text": evaluation["chunk"],
                "is_relevant": evaluation["is_relevant"]
            })

        return {
            "question": question,
            "expected_answer": expected_answer,
            "total_chunks": len(chunks),
            "is_error": False,
            "retrieved_chunks": retrieved_chunks,  # Keep for internal use (calculate_map, etc.)
            "detailed_chunks": detailed_chunks,  # Consolidated chunk info for JSONL logging
            "relevant_ranks": relevant_ranks,
            "average_precision": average_precision,
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": reciprocal_rank,
            "map_data": {
                "relevant_ranks": relevant_ranks,
                "average_precision": average_precision,
                "first_relevant_rank": first_relevant_rank,
            },
        }, core_time

    def calculate_map(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Average Precision from evaluation results."""

        average_precisions = []
        for result in evaluation_results:
            if "average_precision" in result:
                average_precisions.append(result["average_precision"])

        if not average_precisions:
            return 0.0

        return np.mean(average_precisions)

    def calculate_mrr(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Reciprocal Rank from evaluation results."""
        
        reciprocal_ranks = []
        for result in evaluation_results:
            if "first_relevant_rank" in result:
                first_rank = result["first_relevant_rank"]
                if first_rank is not None and first_rank > 0:
                    reciprocal_ranks.append(1.0 / first_rank)
                else:
                    reciprocal_ranks.append(0.0)
        
        if not reciprocal_ranks:
            return 0.0
        
        return np.mean(reciprocal_ranks)

    def evaluate_dataset(
        self,
        examples: List[Dict[str, Any]],
        document_dir: Path,
        rag_config: Dict[str, Any],
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance on a dataset of examples.

        Args:
            examples: List of evaluation examples
            document_dir: Directory containing PDF documents
            rag_config: RAG configuration parameters
            max_examples: Maximum number of examples to evaluate (for testing)

        Returns:
            Dictionary containing evaluation results and metrics
        """

        # Create RAG agent with specified configuration
        rag_agent = self.create_rag_agent(**rag_config)
        # Store reference for cleanup
        self._current_rag_agent = rag_agent

        # Limit examples if specified
        if max_examples:
            examples = examples[:max_examples]

        evaluation_results = []
        processed_docs = {}  # Cache processed documents
        example_times = []  # Track timing per example

        # Create progress bar for examples
        pbar = tqdm(examples, desc="Processing examples", unit="example")

        for i, example in enumerate(pbar):
            pbar.set_description(f"Processing {example['doc_id'][:20]}...")
            
            # Load document if not cached
            doc_id = example["doc_id"]
            doc_ingestion_time = 0.0
            
            if doc_id not in processed_docs:
                # Start timing for document ingestion only
                doc_start_time = time.time()
                
                doc_path = document_dir / doc_id
                if doc_path.exists():
                    try:
                        document_text = rag_agent.document_processor.file_to_text(
                            doc_path
                        )
                        processed_docs[doc_id] = document_text
                        doc_ingestion_time = time.time() - doc_start_time
                    except Exception as e:
                        tqdm.write(f"Error processing document {doc_id}: {e}")
                        continue
                else:
                    tqdm.write(f"Document not found: {doc_path}")
                    continue

            document_text = processed_docs[doc_id]

            # Evaluate retrieval for this example
            try:
                result, core_retrieval_time = self.evaluate_retrieval_for_example(
                    example, document_text, rag_agent
                )
                result["doc_id"] = doc_id
                result["example_index"] = i
                evaluation_results.append(result)

                # Record timing for core operations only (ingestion + retrieval + reranking)
                total_core_time = doc_ingestion_time + core_retrieval_time
                example_times.append(total_core_time)

            except Exception as e:
                tqdm.write(f"Error evaluating example {i+1}: {e}")
                traceback.print_exc()
                # For failed examples, still record the document ingestion time
                example_times.append(doc_ingestion_time)
                continue

            # Clean up memory after each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Calculate overall metrics
        map_score = self.calculate_map(evaluation_results)
        mrr_score = self.calculate_mrr(evaluation_results)

        # Additional metrics
        total_examples = len(evaluation_results)
        examples_with_relevant = len(
            [r for r in evaluation_results if r.get("average_precision", 0) > 0]
        )

        # Calculate timing metrics
        avg_time_per_example = (
            sum(example_times) / len(example_times) if example_times else 0.0
        )

        metrics = {
            "mean_average_precision": map_score,
            "mean_reciprocal_rank": mrr_score,
            "total_examples_evaluated": total_examples,
            "examples_with_relevant_chunks": examples_with_relevant,
            "success_rate": (
                examples_with_relevant / total_examples if total_examples > 0 else 0.0
            ),
            "avg_time_per_example": avg_time_per_example,
            "rag_config": rag_config,
        }

        # Clean up GPU memory after evaluating this configuration
        # Move models to CPU to truly free GPU memory
        rag_agent.move_models_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "metrics": metrics,
            "evaluation_results": evaluation_results,
            "config": rag_config,
        }
    
    def log_detailed_result_to_jsonl(
        self, 
        result: Dict[str, Any], 
        config_name: str, 
        config_id: int, 
        jsonl_file: Path
    ):
        """
        Log detailed evaluation result to JSONL file.
        
        Args:
            result: Evaluation result from evaluate_retrieval_for_example
            config_name: Name of the configuration being evaluated
            config_id: ID of the configuration
            jsonl_file: Path to JSONL output file
        """
        
        # Prepare JSONL entry with essential information only
        jsonl_entry = {
            "config_id": config_id,
            "config_name": config_name,
            "is_error": result.get("is_error", False),
            "error": result.get("error", ""),
            "doc_id": result.get("doc_id", ""),
            "example_index": result.get("example_index", -1),
            "question": result.get("question", ""),
            "expected_answer": result.get("expected_answer", ""),
            "total_chunks": result.get("total_chunks", 0),
            "retrieved_chunks": result.get("detailed_chunks", []),  # Only the consolidated chunk info
            "relevant_ranks": result.get("relevant_ranks", []),
            "average_precision": result.get("average_precision", 0.0),
            "first_relevant_rank": result.get("first_relevant_rank"),
            "reciprocal_rank": result.get("reciprocal_rank", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to JSONL file
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")

    def cleanup_current_agent(self):
        """Clean up the current RAG agent and free GPU memory."""
        if self._current_rag_agent:
            try:
                self._current_rag_agent.move_models_to_cpu()
            except Exception as e:
                print(f"Warning: Could not move models to CPU: {e}")
            finally:
                self._current_rag_agent = None


def load_evaluation_configs(
    suite_file: str = "config/evaluation_suite.yaml",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load evaluation configurations from YAML files."""

    # Load the main suite configuration
    with open(suite_file, "r") as f:
        suite_config = yaml.safe_load(f)

    configs = []
    config_dir = Path("config")

    # Load each individual config file
    for config_file in suite_config["configs"]:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                configs.append(config)
        else:
            print(f"Warning: Config file {config_file} not found, skipping...")

    return configs, suite_config["settings"]


def main():
    """Main evaluation function."""

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    # Load filtered examples
    with open("text_only_eval_examples.json", "r") as f:
        examples = json.load(f)

    # Examples loaded - progress will be shown via tqdm

    # Note: This evaluation assumes you have the PDF documents available
    # For demonstration, we'll create a mock evaluation with a subset
    document_dir = Path("documents")  # Adjust as needed

    # Load RAG configurations from YAML files
    # You can specify a different suite file as command line argument
    import sys

    suite_file = sys.argv[1] if len(sys.argv) > 1 else "config/evaluation_suite.yaml"

    try:
        rag_configs, settings = load_evaluation_configs(suite_file)
        print(f"Loaded {len(rag_configs)} configurations from {suite_file}")
    except Exception as e:
        print(f"Error loading configurations: {e}")
        return

    # Initialize evaluator
    judge_workers = int(os.getenv("JUDGE_WORKERS", "4"))
    evaluator = RetrievalEvaluator(judge_workers=judge_workers)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_csv = settings.get("csv_filename", "retrieval_evaluation_results.csv")
    base_json = settings.get("json_filename", "retrieval_evaluation_results.json")
    base_jsonl = settings.get("jsonl_filename", "retrieval_evaluation_detailed.jsonl")

    # Add timestamp to filenames and put in output folder
    csv_name = base_csv.replace(".csv", f"_{timestamp}.csv")
    json_name = base_json.replace(".json", f"_{timestamp}.json")
    jsonl_name = base_jsonl.replace(".jsonl", f"_{timestamp}.jsonl")
    csv_file = output_dir / csv_name
    json_file = output_dir / json_name
    jsonl_file = output_dir / jsonl_name

    # Run evaluations
    all_results = []
    max_examples = settings.get("max_examples", None)

    # Initialize CSV file with headers
    csv_headers = [
        "config_id",
        "config_name",
        "chunk_size",
        "retrieval_method",
        "retrieval_model",
        "reranker_method",
        "reranker_model",
        "processor_class",
        "k1",
        "b",
        "mean_average_precision",
        "mean_reciprocal_rank",
        "success_rate",
        "total_examples_evaluated",
        "examples_with_relevant_chunks",
        "avg_time_per_example",
    ]

    # Write headers if file doesn't exist
    if not Path(csv_file).exists():
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # Create progress bar for configurations
    config_pbar = tqdm(rag_configs, desc="Evaluating configs", unit="config")

    for i, config in enumerate(config_pbar):
        config_name = config.get("name", f"Config_{i+1}")
        config_pbar.set_description(f"Config: {config_name[:30]}...")

        try:
            # Create a clean config without display fields for RAG agent
            rag_config = {k: v for k, v in config.items() if k != "name"}

            results = evaluator.evaluate_dataset(
                examples=examples,
                document_dir=document_dir,
                rag_config=rag_config,
                max_examples=max_examples,
            )

            all_results.append(results)
            
            # Log detailed results to JSONL file
            for result in results["evaluation_results"]:
                evaluator.log_detailed_result_to_jsonl(
                    result=result,
                    config_name=config_name,
                    config_id=i + 1,
                    jsonl_file=jsonl_file
                )
            
            # Update progress bar with results
            metrics = results["metrics"]
            tqdm.write(
                f"{config_name}: MAP={metrics['mean_average_precision']:.4f}, MRR={metrics['mean_reciprocal_rank']:.4f}, Success={metrics['success_rate']:.4f}"
            )

            # Append results to CSV
            csv_row = [
                i + 1,  # config_id
                config.get("name", f"Config_{i+1}"),  # config_name
                config["chunk_size"],
                config["retrieval_method"],
                config.get("retrieval_kwargs", {}).get("model_name", ""),
                config.get("reranker_method", ""),
                config.get("reranker_kwargs", {}).get("model_name", ""),
                config.get("processor_class", "PyPDF2Processor"),  # processor_class
                config.get("retrieval_kwargs", {}).get("k1", ""),
                config.get("retrieval_kwargs", {}).get("b", ""),
                metrics["mean_average_precision"],
                metrics["mean_reciprocal_rank"],
                metrics["success_rate"],
                metrics["total_examples_evaluated"],
                metrics["examples_with_relevant_chunks"],
                metrics["avg_time_per_example"],
            ]

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

        except Exception as e:

            tqdm.write(f"Error evaluating {config_name}: {e}")
            traceback.print_exc()
        
        finally:
            # Clean up GPU memory after each configuration evaluation
            # This is important because different configs may use different models
            evaluator.cleanup_current_agent()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Save results
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - CSV: {csv_file}")
    print(f"  - JSONL (detailed): {jsonl_file}")


if __name__ == "__main__":
    main()
