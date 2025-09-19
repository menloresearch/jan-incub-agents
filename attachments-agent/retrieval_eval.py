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
from processors import PyPDF2Processor, MarkItDownProcessor, DoclingProcessor, NativeProcessor
from utils import split_into_multi_chunks
from llm_wrapper import OpenAIApiWrapper


class RetrievalEvaluator:
    """Evaluator for retrieval + reranking systems."""

    def __init__(
        self,
        judge_model: str = "gpt-5-mini",
        base_url: Optional[str] = None,
    ):
        self.model_id = "gpt-4o"  # just for tokenizer
        self.judge_model = judge_model
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        
        # Mapping of processor class names to actual classes
        self.processor_classes = {
            "PyPDF2Processor": PyPDF2Processor,
            "MarkItDownProcessor": MarkItDownProcessor,
            "DoclingProcessor": DoclingProcessor,
            "NativeProcessor": NativeProcessor,
        }

        # Initialize LLM judge for chunk relevance evaluation
        self.judge_llm = OpenAIApiWrapper(
            model_id=self.judge_model,
            base_url=self.base_url,
            sampling_params={
                "temperature": 0.1,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
        )

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

    def judge_chunk_relevance(
        self, question: str, chunk: str, expected_answer: str
    ) -> Dict[str, Any]:
        """
        Use LLM judge to evaluate if a chunk is relevant to answering the question.

        Returns:
            Dict with 'is_relevant' (bool), 'confidence' (float), 'reasoning' (str)
        """

        judge_prompt = f"""
You are an expert judge evaluating the relevance of text chunks for answering questions.

Question: {question}
Expected Answer: {expected_answer}

Text Chunk to Evaluate:
{chunk}

Task: Determine if this text chunk contains information that would help answer the question correctly.

Consider:
1. Does the chunk contain facts, data, or context relevant to the question?
2. Would this chunk help generate the expected answer or something very similar?
3. Is the information in the chunk directly or indirectly related to what's being asked?

Respond with valid JSON in this exact format:
{{
  "is_relevant": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this chunk is or isn't relevant"
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

                return {
                    "is_relevant": bool(result.get("is_relevant", False)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": str(result.get("reasoning", "No reasoning provided")),
                    "raw_response": response,
                }

            except Exception as e:
                print(f"Failed to parse/repair JSON response: {e}")
                print(f"Raw response: {response}")

                # Fallback to default values
                return {
                    "is_relevant": False,
                    "confidence": 0.0,
                    "reasoning": f"Failed to parse JSON response: {str(e)}",
                    "raw_response": response,
                }

        except Exception as e:
            print(f"Error in judge evaluation: {e}")
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "reasoning": f"Error in evaluation: {str(e)}",
                "raw_response": "",
            }

    def evaluate_retrieval_for_example(
        self,
        example: Dict[str, Any],
        document_text: str,
        rag_agent: SimpleRAGDocumentAgent,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance for a single example.

        Returns evaluation metrics including MRR-related data.
        """

        question = example["question"]
        expected_answer = example["answer"]

        # Split document into chunks (same as RAG agent)
        chunks = split_into_multi_chunks(document_text, chunk_size=rag_agent.chunk_size)

        if not chunks:
            return {
                "question": question,
                "expected_answer": expected_answer,
                "error": "No chunks generated from document",
                "retrieved_chunks": [],
                "relevance_scores": [],
                "map_data": {"relevant_ranks": [], "average_precision": 0.0},
            }

        # Retrieve chunks using the RAG agent's retrieval system
        retrieved_chunks = rag_agent.retrieval.retrieve(
            query=question, documents=chunks, top_k=10  # Retrieve more for evaluation
        )

        # Apply reranking if available
        if rag_agent.reranker:
            retrieved_chunks = rag_agent.reranker.rerank(
                query=question, documents=retrieved_chunks, top_k=10
            )

        # Judge relevance of each retrieved chunk
        relevance_evaluations = []
        relevant_ranks = []

        for rank, chunk in enumerate(retrieved_chunks, 1):
            judge_result = self.judge_chunk_relevance(question, chunk, expected_answer)
            relevance_evaluations.append(
                {
                    "rank": rank,
                    "chunk": chunk,
                    "is_relevant": judge_result["is_relevant"],
                    "confidence": judge_result["confidence"],
                    "reasoning": judge_result["reasoning"],
                }
            )

            if judge_result["is_relevant"]:
                relevant_ranks.append(rank)

        # Calculate average precision (for MAP)
        average_precision = 0.0
        if relevant_ranks:
            precision_sum = 0.0
            for i, rank in enumerate(relevant_ranks):
                precision_at_rank = (i + 1) / rank
                precision_sum += precision_at_rank
            average_precision = precision_sum / len(relevant_ranks)

        return {
            "question": question,
            "expected_answer": expected_answer,
            "total_chunks": len(chunks),
            "retrieved_chunks": retrieved_chunks,
            "relevance_evaluations": relevance_evaluations,
            "relevant_ranks": relevant_ranks,
            "average_precision": average_precision,
            "map_data": {
                "relevant_ranks": relevant_ranks,
                "average_precision": average_precision,
                "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
            },
        }

    def calculate_map(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Average Precision from evaluation results."""

        average_precisions = []
        for result in evaluation_results:
            if "average_precision" in result:
                average_precisions.append(result["average_precision"])

        if not average_precisions:
            return 0.0

        return np.mean(average_precisions)

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
            start_time = time.time()  # Start timing this example

            # Load document if not cached
            doc_id = example["doc_id"]
            if doc_id not in processed_docs:
                doc_path = document_dir / doc_id
                if doc_path.exists():
                    try:
                        document_text = rag_agent.document_processor.file_to_text(
                            doc_path
                        )
                        processed_docs[doc_id] = document_text
                    except Exception as e:
                        tqdm.write(f"Error processing document {doc_id}: {e}")
                        continue
                else:
                    tqdm.write(f"Document not found: {doc_path}")
                    continue

            document_text = processed_docs[doc_id]

            # Evaluate retrieval for this example
            try:
                result = self.evaluate_retrieval_for_example(
                    example, document_text, rag_agent
                )
                result["doc_id"] = doc_id
                result["example_index"] = i
                evaluation_results.append(result)

                # Record timing for this example
                end_time = time.time()
                example_times.append(end_time - start_time)

            except Exception as e:
                tqdm.write(f"Error evaluating example {i+1}: {e}")
                # Still record timing even for failed examples
                end_time = time.time()
                example_times.append(end_time - start_time)
                continue

            # Clean up memory after each iteration
            if hasattr(rag_agent.retrieval, "model"):
                rag_agent.retrieval.model.cpu()
            if rag_agent.reranker and hasattr(rag_agent.reranker, "model"):
                rag_agent.reranker.model.cpu()
            gc.collect()

        # Calculate overall metrics
        map_score = self.calculate_map(evaluation_results)

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
            "total_examples_evaluated": total_examples,
            "examples_with_relevant_chunks": examples_with_relevant,
            "success_rate": (
                examples_with_relevant / total_examples if total_examples > 0 else 0.0
            ),
            "avg_time_per_example": avg_time_per_example,
            "rag_config": rag_config,
        }

        return {
            "metrics": metrics,
            "evaluation_results": evaluation_results,
            "config": rag_config,
        }


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
    evaluator = RetrievalEvaluator()

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_csv = settings.get("csv_filename", "retrieval_evaluation_results.csv")
    base_json = settings.get("json_filename", "retrieval_evaluation_results.json")

    # Add timestamp to filenames and put in output folder
    csv_name = base_csv.replace(".csv", f"_{timestamp}.csv")
    json_name = base_json.replace(".json", f"_{timestamp}.json")
    csv_file = output_dir / csv_name
    json_file = output_dir / json_name

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
            # Update progress bar with results
            metrics = results["metrics"]
            tqdm.write(
                f"{config_name}: MAP={metrics['mean_average_precision']:.4f}, Success={metrics['success_rate']:.4f}"
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
                metrics["success_rate"],
                metrics["total_examples_evaluated"],
                metrics["examples_with_relevant_chunks"],
                metrics["avg_time_per_example"],
            ]

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)

        except Exception as e:
            import traceback

            tqdm.write(f"Error evaluating {config_name}: {e}")
            traceback.print_exc()

    # Save results
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - CSV: {csv_file}")


if __name__ == "__main__":
    main()
