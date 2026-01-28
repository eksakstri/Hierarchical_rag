# Hierarchical_rag

A lightweight, open-source NLP application that converts unstructured text paragraphs into structured form data using chunking + extractive Question Answering (QA) with rule-based constraints.
Built entirely in Python using Hugging Face Transformers (no paid APIs).

# Problem Statement

Unstructured text (emails, reports, descriptions) often contains important information that needs to be captured in fixed form fields.

This project solves that by:
Breaking text into manageable chunks
Asking targeted questions per field
Applying rules to ensure factual, non-inferential extraction
Returning structured JSON output

# WorkFlow

Input paragraphs 
      →
Tokenization
      →
Re-grouping to big similar meaning chunks
      →
Tree Structure
      →
Rule Injection (System Prompt Simulation)
      →
Transformer-based QA Model
      →
Confidence Filtering
      →
Rule-based Post Processing
      →
Structured Output (JSON)

