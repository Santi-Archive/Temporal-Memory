# Temporal Memory Layer

A comprehensive system that processes long-form stories and converts them into structured, queryable memory systems. The system consists of 5 major stages that transform raw narrative text into machine-readable temporal memory.

## Overview

The Temporal Memory Layer processes narrative text through the following stages:

1. **Text Processing** - Chapter segmentation, sentence tokenization, and linguistic annotation
2. **Event & Role Extraction** - Semantic Role Labeling (SRL) to extract event frames
3. **Temporal Normalization** - Converts time expressions into standardized timestamps
4. **Semantic Representation** - Generates embeddings for semantic similarity
5. **Memory Storage** - Combines all data into a unified JSON memory module

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install `huggingface_hub` (required for model downloads):
```bash
pip install huggingface_hub
```

3. Download all models locally (recommended for faster runtime):
```bash
python download_models.py
```

This will download all required models (~600MB total) directly into the `models/` directory in your project.

**Model Storage:**
- Sentence Transformers: `models/sentence_transformers/` (contains `all-MiniLM-L6-v2`)
- HuggingFace: `models/huggingface/` (contains `dbmdz/bert-large-cased-finetuned-conll03-english`)
- spaCy: Installed to spaCy's system location (managed by spaCy)

**Note:** Models are stored locally in the project directory, making them easy to share or version control. The download script uses `huggingface_hub` to download models directly to the specified directories.

## Usage

### Command Line

**Important:** Always use `python -m backend.pipeline` (not `python backend/pipeline.py`) to avoid import errors.

Run the complete pipeline from the command line:

```bash
python -m backend.pipeline <input_file> [output_dir] [reference_date] [embedding_model]
```

**Arguments:**
- `input_file` (required): Path to input text file
- `output_dir` (optional): Output directory (default: `output`)
- `reference_date` (optional): Reference date for normalization in YYYY-MM-DD format (default: current date)
- `embedding_model` (optional): Sentence transformer model name (default: `all-MiniLM-L6-v2`)

**Example:**
```bash
python -m backend.pipeline data/story.txt output 2025-01-15 all-MiniLM-L6-v2
```

⚠️ **Common Error:** If you see `ImportError: attempted relative import with no known parent package`, you're running the file directly. Always use `python -m backend.pipeline` instead.

### Python API

Use the pipeline programmatically:

```python
from backend.pipeline import run_pipeline

# Run the complete pipeline
memory_module_path = run_pipeline(
    input_file="data/story.txt",
    output_dir="output",
    reference_date="2025-01-15",
    embedding_model="all-MiniLM-L6-v2"
)

print(f"Memory module saved to: {memory_module_path}")
```

### Individual Steps

You can also run individual steps:

```python
from backend.step1_text_processing import process_text
from backend.step2_event_extraction import extract_events
from backend.step3_temporal_normalization import normalize_temporal_expressions
from backend.step4_semantic_representation import create_semantic_representations
from backend.step5_memory_storage import create_memory_module

# Step 1: Text Processing
text_data = process_text("data/story.txt", "output")

# Step 2: Event Extraction
events = extract_events("output", "output")

# Step 3: Temporal Normalization
timestamps = normalize_temporal_expressions("output", "output")

# Step 4: Semantic Representation
semantic_data = create_semantic_representations("output", "output")

# Step 5: Memory Storage
memory_module = create_memory_module("output", "output")
```

## Output Structure

The pipeline generates the following output files:

```
output/
├── preprocessed/
│   ├── chapters.json          # Chapter segmentation data
│   └── sentences.json         # Tokenized sentences with annotations
└── memory/
    ├── events.json            # Extracted event frames
    ├── timestamps.json        # Normalized temporal expressions
    ├── event_embeddings.json  # Embedding vectors
    ├── memory_semantic.json   # Semantic memory table
    └── memory_module.json     # Unified memory module (main output)
```

### Memory Module Structure

The main output file `memory_module.json` contains:

```json
{
  "metadata": {
    "total_chapters": 5,
    "total_sentences": 150,
    "total_events": 200,
    "total_characters": 10,
    "total_locations": 5,
    "embedding_dim": 384,
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "chapters": [...],
  "sentences": [...],
  "events": [...],
  "timestamps": {...},
  "embeddings": {...},
  "characters": [...],
  "entities": {...}
}
```

## Event Frame Structure

Each event frame contains:

- `event_id`: Unique identifier
- `chapter_id`: Associated chapter
- `sentence_id`: Source sentence
- `actor`: Subject/ARG0
- `action`: Verb/predicate
- `target`: Object/ARG1/ARG2
- `location`: Location modifier (ARGM-LOC)
- `time_raw`: Original time expression (ARGM-TMP)
- `time_normalized`: Normalized timestamp
- `time_type`: Type of time expression
- `entities`: Named entities
- `roles`: Semantic roles
- `dependencies`: Dependency parse information

## Dependencies

- `spacy>=3.7.0` - Natural language processing (POS tagging, dependency parsing, NER)
- `transformers>=4.35.0` - HuggingFace transformers for Semantic Role Labeling (SRL)
- `torch>=2.0.0` - PyTorch for deep learning models
- `sentence-transformers>=2.2.0` - Sentence embeddings for semantic similarity
- `pandas>=2.0.0` - Data manipulation and semantic memory tables
- `numpy>=1.24.0` - Numerical operations for embeddings
- `python-heideltime>=0.1.0` - Temporal normalization (with fallback)
- `huggingface_hub>=0.20.0` - Model downloading and management

## Project Structure

```
Temporal-Memory/
├── backend/
│   ├── __init__.py                    # Package initialization
│   ├── pipeline.py                    # Main orchestrator - runs all 5 steps
│   ├── step1_text_processing.py       # Step 1: Chapter segmentation, tokenization, annotation
│   ├── step2_event_extraction.py      # Step 2: Event extraction with SRL and dependency parsing
│   ├── step3_temporal_normalization.py # Step 3: Time expression normalization
│   ├── step4_semantic_representation.py # Step 4: Embedding generation and semantic memory
│   ├── step5_memory_storage.py        # Step 5: Unified memory module creation
│   ├── model_cache.py                 # Model cache utilities and status checking
│   └── utils.py                       # Shared utilities (JSON I/O, directory management)
├── data/
│   └── sample_story.txt               # Example story for testing
├── models/                            # Local model storage (created after download_models.py)
│   ├── sentence_transformers/         # Sentence transformer models
│   └── huggingface/                   # HuggingFace transformer models
├── output/                            # Generated output files (created during processing)
│   ├── preprocessed/                  # Step 1 outputs
│   │   ├── chapters.json              # Chapter segmentation data
│   │   └── sentences.json             # Tokenized and annotated sentences
│   └── memory/                        # Steps 2-5 outputs
│       ├── events.json                # Extracted event frames
│       ├── timestamps.json            # Normalized temporal expressions
│       ├── event_embeddings.json      # Embedding vectors
│       ├── memory_semantic.json       # Semantic memory table
│       └── memory_module.json         # Main unified output (50% deliverable)
├── download_models.py                 # Script to download all models locally
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── GUIDE.md                           # Comprehensive guide with detailed explanations
```

## Notes

- The system uses dependency parsing as a fallback when SRL models are unavailable
- Temporal normalization includes a fallback mechanism if HeidelTime is not available
- The default embedding model (`all-MiniLM-L6-v2`) is fast and efficient; use `all-mpnet-base-v2` for higher accuracy
- Input text should be plain text format (.txt) with chapter markers like "Chapter 1", "CHAPTER 2", etc.

## License

This project is part of a thesis implementation for a Temporal Memory Layer system.

