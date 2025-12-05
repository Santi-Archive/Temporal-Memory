to run: python -m backend.pipeline data/sample_story.txt

# Temporal Memory Layer - Comprehensive Guide

## 📖 What is This System?

The **Temporal Memory Layer** is an intelligent system that reads long-form stories (like novels, narratives, or any text with multiple chapters) and converts them into a structured, searchable database. Think of it as a "smart reader" that understands:

-   **Who** did **what** to **whom**
-   **When** events happened
-   **Where** events occurred
-   **How** events relate to each other

The system transforms raw text into a machine-readable format that can be queried, analyzed, and used for consistency checking, timeline analysis, and story understanding.

---

## 🎯 Why Do We Need This?

### The Problem

When analyzing long narratives manually, it's difficult to:

-   Track all events and their relationships
-   Verify timeline consistency
-   Find contradictions in the story
-   Understand character interactions over time
-   Query specific information quickly

### The Solution

This system automatically:

-   ✅ Extracts all events from the text
-   ✅ Identifies characters and their actions
-   ✅ Normalizes time expressions ("yesterday" → "2025-01-14")
-   ✅ Creates searchable embeddings for semantic similarity
-   ✅ Builds a unified memory structure for easy querying

---

## 🔄 How It Works: The 5-Step Process

The system processes stories through five sequential stages. Here's what happens at each step:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Story Text                     │
│              (e.g., "Chapter 1\nJohn walked...")            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   STEP 1: TEXT PROCESSING             │
        │   • Split into chapters                │
        │   • Break into sentences               │
        │   • Identify parts of speech           │
        │   • Extract named entities             │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   STEP 2: EVENT EXTRACTION            │
        │   • Find all actions (verbs)          │
        │   • Identify actors (who)              │
        │   • Identify targets (what/whom)       │
        │   • Extract locations and times        │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   STEP 3: TEMPORAL NORMALIZATION       │
        │   • Convert "yesterday" → "2025-01-14" │
        │   • Standardize time expressions       │
        │   • Handle relative times              │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   STEP 4: SEMANTIC REPRESENTATION     │
        │   • Create vector embeddings          │
        │   • Enable similarity search           │
        │   • Build semantic memory table       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   STEP 5: MEMORY STORAGE              │
        │   • Combine all data                  │
        │   • Create unified memory module      │
        │   • Export to JSON                     │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         OUTPUT: memory_module.json     │
        │    (Structured, queryable memory)      │
        └───────────────────────────────────────┘
```

---

## 📚 Detailed Step-by-Step Explanation

### Step 1: Text Processing

**Purpose:** Prepares raw story text for deeper analysis by breaking it into structured, annotated components.

**What it does:**

-   **Chapter Segmentation**: Identifies and splits text by chapter markers (Chapter 1, CHAPTER 2, etc.)
-   **Sentence Tokenization**: Breaks each chapter into individual sentences using spaCy's sentence boundary detection
-   **Word Tokenization**: Splits sentences into individual words, punctuation, and symbols
-   **Linguistic Annotation**: Performs comprehensive linguistic analysis on each sentence

**Technologies Used:**

-   **spaCy (`en_core_web_sm`)**:
    -   Part-of-Speech (POS) tagging: Identifies nouns, verbs, adjectives, etc.
    -   Dependency parsing: Maps grammatical relationships (subject, object, modifiers)
    -   Named Entity Recognition (NER): Identifies PERSON, LOCATION, DATE, TIME, ORG, etc.
    -   Lemmatization: Converts words to their base forms (walked → walk)

**Example Input:**

```
Chapter 1
John walked to the river yesterday. He saw a beautiful sunset.
```

**Processing Details:**

1. **Chapter Segmentation**: Uses regex patterns to find chapter markers and split text accordingly
2. **Sentence Splitting**: spaCy automatically detects sentence boundaries based on punctuation and context
3. **Token Analysis**: For each word, extracts:
    - Text: The original word
    - Lemma: Base form (walked → walk)
    - POS: Part of speech (NOUN, VERB, ADJ, etc.)
    - Tag: Detailed tag (NNP for proper noun, VBD for past tense verb)
    - Dependency: Grammatical role (nsubj = subject, dobj = direct object)
    - Head: The word this token depends on
4. **Entity Extraction**: Identifies and labels named entities with their types

**Output Files Explained:**

#### `output/preprocessed/chapters.json`

Contains chapter-level information:

```json
{
    "chapters": [
        {
            "chapter_id": "chapter_1",
            "chapter_number": "1",
            "title": "Chapter 1",
            "text": "Full chapter text content..."
        }
    ],
    "total_chapters": 5,
    "total_sentences": 150
}
```

**Fields:**

-   `chapter_id`: Unique identifier (chapter_1, chapter_2, etc.)
-   `chapter_number`: Chapter number as found in text
-   `title`: Chapter title/marker
-   `text`: Full text content of the chapter

#### `output/preprocessed/sentences.json`

Contains sentence-level annotations:

```json
{
  "sentences": [
    {
      "sentence_id": "sentence_1",
      "chapter_id": "chapter_1",
      "text": "John walked to the river yesterday.",
      "tokens": [
        {
          "text": "John",
          "lemma": "John",
          "pos": "PROPN",
          "tag": "NNP",
          "dep": "nsubj",
          "head": "walked",
          "head_pos": "VERB"
        },
        // ... more tokens
      ],
      "pos_tags": [...],
      "dependencies": [...],
      "entities": [
        {
          "text": "John",
          "label": "PERSON",
          "start": 0,
          "end": 4
        }
      ]
    }
  ],
  "total_sentences": 150
}
```

**Fields:**

-   `sentence_id`: Unique identifier (sentence_1, sentence_2, etc.)
-   `chapter_id`: Links sentence to its chapter
-   `text`: Original sentence text
-   `tokens`: Array of all words with full linguistic information
-   `pos_tags`: Parts of speech for each token
-   `dependencies`: Dependency parse tree showing grammatical relationships
-   `entities`: Named entities found in the sentence (PERSON, LOCATION, DATE, etc.)

---

### Step 2: Event & Role Extraction

**Purpose:** Transforms linguistic annotations into structured event frames that capture who did what to whom, when, and where.

**What it does:**

-   **Predicate Identification**: Finds all verbs (actions) in each sentence
-   **Semantic Role Labeling (SRL)**: Attempts to extract semantic roles using HuggingFace transformers
-   **Dependency-Based Extraction**: Uses spaCy dependency parsing to fill gaps when SRL is unavailable
-   **Event Frame Construction**: Creates structured event objects with all extracted information

**Technologies Used:**

-   **HuggingFace Transformers** (`dbmdz/bert-large-cased-finetuned-conll03-english`):
    -   Token classification model for Semantic Role Labeling
    -   Identifies semantic roles: ARG0 (actor), ARG1/ARG2 (target), ARGM-LOC (location), ARGM-TMP (time)
    -   Falls back to dependency parsing if model unavailable
-   **spaCy Dependency Parsing**:
    -   Extracts subject (nsubj) → Actor
    -   Extracts direct object (dobj) → Target
    -   Extracts prepositional phrases (prep + pobj) → Location/Time
    -   Handles complex grammatical structures

**Processing Details:**

1. **Verb Extraction**: Scans each sentence's POS tags to find all verbs
2. **Multiple Events**: One sentence can produce multiple events if it contains multiple verbs
3. **Role Extraction**:
    - **Actor (ARG0)**: Subject of the verb (who performs the action)
    - **Action (V)**: The verb/predicate itself
    - **Target (ARG1/ARG2)**: Direct/indirect object (what/who is affected)
    - **Location (ARGM-LOC)**: Prepositional phrases indicating location
    - **Time (ARGM-TMP)**: Temporal modifiers (yesterday, in the afternoon, etc.)
4. **Entity Integration**: Includes named entities from Step 1

**Example:**
From the sentence "John walked to the river yesterday", Step 2 creates:

```json
{
    "event_id": "event_1",
    "chapter_id": "chapter_1",
    "sentence_id": "sentence_1",
    "actor": "John",
    "action": "walked",
    "action_lemma": "walk",
    "target": "river",
    "location": "river",
    "time_raw": "yesterday",
    "entities": ["John", "river"],
    "roles": {
        "ARG0": "John",
        "ARG1": "river",
        "ARGM-LOC": "river",
        "ARGM-TMP": "yesterday"
    },
    "dependencies": [...]
}
```

**Output File Explained:**

#### `output/memory/events.json`

Contains all extracted event frames:

```json
{
  "events": [
    {
      "event_id": "event_1",
      "chapter_id": "chapter_1",
      "sentence_id": "sentence_1",
      "actor": "John",
      "action": "walked",
      "action_lemma": "walk",
      "target": "river",
      "location": "river",
      "time_raw": "yesterday",
      "entities": ["John", "river"],
      "roles": {...},
      "dependencies": [...]
    }
  ],
  "total_events": 200
}
```

**Event Frame Fields:**

-   `event_id`: Unique identifier (event_1, event_2, etc.)
-   `chapter_id`: Source chapter
-   `sentence_id`: Source sentence
-   `actor`: Subject/ARG0 (who performed the action)
-   `action`: Verb/predicate (the action itself)
-   `action_lemma`: Base form of the verb
-   `target`: Object/ARG1/ARG2 (what/who was affected)
-   `location`: Location modifier/ARGM-LOC (where it happened)
-   `time_raw`: Original time expression/ARGM-TMP (when it happened)
-   `entities`: Named entities from the sentence
-   `roles`: Dictionary of semantic roles (ARG0, ARG1, ARGM-LOC, ARGM-TMP, etc.)
-   `dependencies`: Full dependency parse information

---

### Step 3: Temporal Normalization

**Purpose:** Converts vague, relative, or ambiguous time expressions into standardized, machine-readable timestamps.

**What it does:**

-   **Time Expression Collection**: Gathers all time expressions from event frames (ARGM-TMP values)
-   **Normalization**: Converts relative times to absolute dates when possible
-   **Standardization**: Creates consistent time formats for analysis
-   **Placeholder Preservation**: Maintains relative time markers for future resolution

**Technologies Used:**

-   **HeidelTime** (primary, if available):
    -   Advanced temporal expression recognition
    -   Handles complex time expressions
    -   Supports multiple languages and document types
-   **Custom Fallback Normalization** (if HeidelTime unavailable):
    -   Regex-based pattern matching
    -   Rule-based conversion for common expressions
    -   Handles: yesterday, today, tomorrow, "X days later/ago", etc.

**Processing Details:**

1. **Extraction**: Collects all `time_raw` values from events and ARGM-TMP roles
2. **Normalization Strategies**:
    - **Absolute Dates**: "2025-01-15" → kept as-is
    - **Relative to Reference Date**: "yesterday" → calculated from reference date
    - **Time of Day**: "afternoon" → "T-AFTERNOON"
    - **Relative Placeholders**: "two weeks later" → "REL-2W"
3. **Reference Date**: Uses provided reference date (defaults to current date) for calculations
4. **Event Updates**: Adds `time_normalized` and `time_type` fields to each event

**Normalization Examples:**

| Original Expression | Normalized Output | Type     | Notes                             |
| ------------------- | ----------------- | -------- | --------------------------------- |
| "yesterday"         | "2025-01-14"      | DATE     | Calculated from reference date    |
| "two days later"    | "2025-01-17"      | DATE     | Relative calculation              |
| "in the afternoon"  | "T-AFTERNOON"     | TIME     | Time of day marker                |
| "next month"        | "REL-1M"          | RELATIVE | Placeholder for future resolution |
| "2025-01-15"        | "2025-01-15"      | DATE     | Already absolute                  |

**Why this matters:**

-   **Timeline Analysis**: Enables chronological sorting and timeline visualization
-   **Consistency Checking**: Helps detect temporal contradictions (e.g., event A happens after event B, but normalized times show otherwise)
-   **Query Capabilities**: Allows queries like "all events in January 2025"
-   **Relative Resolution**: Placeholders can be resolved when story context is fully understood

**Output Files Explained:**

#### `output/memory/timestamps.json`

Contains normalized time data:

```json
{
    "reference_date": "2025-01-15",
    "normalized_times": {
        "yesterday": {
            "original": "yesterday",
            "normalized": "2025-01-14",
            "time_type": "DATE",
            "confidence": 0.8
        },
        "two days later": {
            "original": "two days later",
            "normalized": "2025-01-17",
            "time_type": "DATE",
            "confidence": 0.8
        }
    },
    "total_expressions": 50
}
```

**Fields:**

-   `reference_date`: Date used for relative time calculations
-   `normalized_times`: Dictionary mapping original expressions to normalized data
-   `total_expressions`: Number of unique time expressions found

**Updated Event Fields:**
After Step 3, events also contain:

-   `time_normalized`: Standardized timestamp
-   `time_type`: Type of time expression (DATE, TIME, RELATIVE, UNKNOWN)
-   `time_confidence`: Confidence score (0.0-1.0) for the normalization

---

### Step 4: Semantic Representation

**Purpose:** Converts symbolic event frames into numerical embeddings that capture semantic meaning, enabling similarity search and semantic analysis.

**What it does:**

-   **Event String Formatting**: Creates human-readable semantic representations of events
-   **Embedding Generation**: Converts event strings into dense vector representations
-   **Semantic Memory Table**: Builds a structured table linking events to their embeddings
-   **Similarity Enabling**: Makes it possible to find semantically similar events

**Technologies Used:**

-   **Sentence Transformers** (`all-MiniLM-L6-v2` by default):
    -   Pre-trained transformer model optimized for semantic similarity
    -   Generates 384-dimensional embeddings
    -   Fast and efficient (good balance of speed and accuracy)
    -   Alternative: `all-mpnet-base-v2` for higher accuracy (768 dimensions)
-   **NumPy**: Numerical operations for embedding vectors
-   **Pandas**: Data structure for semantic memory table

**Processing Details:**

1. **Event String Creation**: Formats each event as:
    ```
    "Actor: {actor}; Action: {action}; Target: {target}; Location: {location}; Time: {time_normalized}."
    ```
2. **Embedding Generation**:
    - Processes all event strings through the sentence transformer model
    - Generates fixed-size vectors (384 dimensions for all-MiniLM-L6-v2)
    - Similar events produce similar vectors (measured by cosine similarity)
3. **Memory Table Construction**: Creates a DataFrame with event metadata and embeddings

**Example Event String:**

From an event frame, creates:

```
"Actor: John; Action: walked; Target: river; Location: river; Time: 2025-01-14."
```

**Embeddings Explained:**

-   **Dimensionality**: Each event becomes a 384-dimensional vector (array of 384 numbers)
-   **Semantic Similarity**: Events with similar meanings have vectors that are close in vector space
-   **Use Cases**:
    -   Find events similar to "John walking to a location"
    -   Cluster events by semantic similarity
    -   Detect recurring patterns in the narrative
    -   Query by semantic meaning rather than exact text match

**Output Files Explained:**

#### `output/memory/event_embeddings.json`

Contains embedding vectors for all events:

```json
{
  "event_ids": ["event_1", "event_2", ...],
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // 384 numbers for event_1
    [0.234, -0.567, 0.890, ...],  // 384 numbers for event_2
    ...
  ],
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2"
}
```

**Fields:**

-   `event_ids`: List of event IDs in the same order as embeddings
-   `embeddings`: Array of arrays, each inner array is a 384-dimensional vector
-   `embedding_dim`: Dimension of the embedding vectors (384 for all-MiniLM-L6-v2)
-   `model_name`: Name of the sentence transformer model used

**Note:** Full embedding vectors are stored here. The memory module contains metadata only.

#### `output/memory/memory_semantic.json`

Contains semantic memory table with event information and embeddings:

```json
{
  "events": [
    {
      "event_id": "event_1",
      "chapter_id": "chapter_1",
      "sentence_id": "sentence_1",
      "semantic_string": "Actor: John; Action: walked; Target: river; Time: 2025-01-14.",
      "normalized_timestamp": "2025-01-14",
      "timestamp_type": "DATE",
      "embedding_vector": [0.123, -0.456, 0.789, ...],
      "embedding_dim": 384
    }
  ],
  "total_events": 200,
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2"
}
```

**Fields:**

-   `events`: Array of event records with semantic information
-   `semantic_string`: Human-readable event representation
-   `normalized_timestamp`: Standardized time from Step 3
-   `embedding_vector`: Full embedding vector (384 numbers)
-   `embedding_dim`: Dimension of embeddings
-   `model_name`: Model used for generation

---

### Step 5: Memory Storage

**Purpose:** Combines all processed data from previous steps into a unified, queryable memory module that serves as the final deliverable.

**What it does:**

-   **Data Aggregation**: Loads and combines outputs from all previous steps
-   **Entity Extraction**: Identifies and catalogs unique characters, locations, organizations, dates, and times
-   **Unified Structure**: Creates a single JSON file containing all information
-   **Metadata Generation**: Produces summary statistics about the processed story

**Technologies Used:**

-   **JSON**: Standard format for data serialization
-   **Python dictionaries/lists**: Data structures for organization
-   **Entity aggregation**: Combines entities from events and sentences

**Processing Details:**

1. **Data Loading**: Reads all previous outputs:
    - Chapters from `chapters.json`
    - Sentences from `sentences.json`
    - Events from `events.json` (with normalized times)
    - Timestamps from `timestamps.json`
    - Embeddings metadata from `event_embeddings.json`
2. **Character Extraction**:
    - Collects all actors from events
    - Extracts PERSON entities from sentences
    - Removes duplicates
3. **Entity Categorization**:
    - **Characters**: All PERSON entities
    - **Locations**: GPE and LOC entities
    - **Organizations**: ORG entities
    - **Dates/Times**: Temporal entities
4. **Metadata Calculation**: Computes totals and statistics

**Final Output Explained:**

#### `output/memory/memory_module.json`

**This is the main output file for the 50% prototype deliverable!**

Complete structure:

```json
{
  "metadata": {
    "total_chapters": 15,
    "total_sentences": 3540,
    "total_events": 5746,
    "total_characters": 146,
    "total_locations": 20,
    "total_organizations": 26,
    "embedding_dim": 384,
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "chapters": [
    {
      "chapter_id": "chapter_1",
      "chapter_number": "1",
      "title": "Chapter 1",
      "text": "Full chapter text..."
    }
  ],
  "sentences": [
    {
      "sentence_id": "sentence_1",
      "chapter_id": "chapter_1",
      "text": "Sentence text...",
      "tokens": [...],
      "entities": [...]
    }
  ],
  "events": [
    {
      "event_id": "event_1",
      "chapter_id": "chapter_1",
      "sentence_id": "sentence_1",
      "actor": "John",
      "action": "walked",
      "target": "river",
      "location": "river",
      "time_raw": "yesterday",
      "time_normalized": "2025-01-14",
      "time_type": "DATE",
      "entities": [...],
      "roles": {...}
    }
  ],
  "timestamps": {
    "reference_date": "2025-01-15",
    "normalized_times": {...},
    "total_expressions": 50
  },
  "embeddings": {
    "event_ids": ["event_1", "event_2", ...],
    "embedding_dim": 384,
    "model_name": "all-MiniLM-L6-v2",
    "note": "Full embedding vectors stored in event_embeddings.json"
  },
  "characters": ["John", "Mary", "Rafe", ...],
  "entities": {
    "locations": ["river", "Riverside", "Manila", ...],
    "organizations": ["Philippine Armed Forces", ...],
    "dates": ["2025-01-14", ...],
    "times": ["T-AFTERNOON", ...],
    "other": [...]
  }
}
```

**Section Explanations:**

1. **`metadata`**: Summary statistics about the processed story

    - Quick overview of story size and complexity
    - Useful for understanding the scope of the narrative

2. **`chapters`**: Complete chapter data from Step 1

    - Full text and structure
    - Useful for chapter-level queries

3. **`sentences`**: Complete sentence data from Step 1

    - All linguistic annotations
    - Useful for sentence-level analysis

4. **`events`**: All event frames from Steps 2-3

    - Complete event information with normalized times
    - **This is the core data for rule checking and timeline analysis**

5. **`timestamps`**: Time normalization data from Step 3

    - Reference date and normalized time expressions
    - Useful for temporal analysis

6. **`embeddings`**: Embedding metadata from Step 4

    - Points to full embeddings in `event_embeddings.json`
    - Includes model information

7. **`characters`**: List of all unique characters

    - Extracted from actors and PERSON entities
    - Useful for character tracking

8. **`entities`**: Categorized entities
    - Locations, organizations, dates, times
    - Useful for entity-based queries

---

## 📁 Complete File Structure & File Explanations

### Project Directory Structure

```
Temporal-Memory/
├── backend/                           # Main processing code
│   ├── __init__.py                    # Package initialization file
│   ├── pipeline.py                    # Main orchestrator - coordinates all 5 steps
│   ├── step1_text_processing.py       # Step 1 implementation
│   ├── step2_event_extraction.py      # Step 2 implementation
│   ├── step3_temporal_normalization.py # Step 3 implementation
│   ├── step4_semantic_representation.py # Step 4 implementation
│   ├── step5_memory_storage.py        # Step 5 implementation
│   ├── model_cache.py                 # Model management utilities
│   └── utils.py                       # Shared helper functions
├── data/                              # Input data directory
│   └── sample_story.txt               # Example story file for testing
├── models/                            # Local model storage (created after download)
│   ├── sentence_transformers/         # Sentence transformer models
│   │   └── (model files: config.json, pytorch_model.bin, etc.)
│   └── huggingface/                   # HuggingFace transformer models
│       └── (model files: config.json, pytorch_model.bin, etc.)
├── output/                            # Generated output files (created during processing)
│   ├── preprocessed/                  # Step 1 outputs
│   │   ├── chapters.json              # Chapter segmentation results
│   │   └── sentences.json             # Tokenized and annotated sentences
│   └── memory/                        # Steps 2-5 outputs
│       ├── events.json                # Extracted event frames
│       ├── timestamps.json            # Normalized temporal expressions
│       ├── event_embeddings.json     # Embedding vectors (large file)
│       ├── memory_semantic.json       # Semantic memory table
│       └── memory_module.json         # Main unified output ⭐
├── download_models.py                 # Script to download models to models/ directory
├── requirements.txt                   # Python package dependencies
├── README.md                          # Quick reference guide
└── GUIDE.md                           # This comprehensive guide
```

### Backend Files Explained

#### `backend/__init__.py`

-   **Purpose**: Python package initialization
-   **What it does**: Makes `backend` a proper Python package, allows imports like `from backend.pipeline import run_pipeline`
-   **Contains**: Package metadata and version information

#### `backend/pipeline.py`

-   **Purpose**: Main orchestrator that runs the complete 5-step pipeline
-   **What it does**:
    -   Validates input file
    -   Sets up output directories
    -   Executes all 5 steps sequentially
    -   Handles errors and provides progress output
    -   Returns path to final memory module
-   **Key Functions**:
    -   `run_pipeline()`: Main function that orchestrates everything
    -   `validate_input()`: Checks if input file exists and is readable
    -   `setup_output_directories()`: Creates output folder structure
    -   `main()`: Command-line entry point
-   **Usage**: Run via `python -m backend.pipeline <input_file>`

#### `backend/step1_text_processing.py`

-   **Purpose**: Implements Step 1 - Text Processing
-   **What it does**:
    -   Segments text into chapters
    -   Tokenizes sentences and words
    -   Performs linguistic annotation (POS, dependencies, NER)
-   **Key Functions**:
    -   `segment_chapters()`: Splits text by chapter markers
    -   `tokenize_sentences()`: Breaks chapters into sentences with spaCy
    -   `annotate_linguistics()`: Extracts POS, dependencies, entities
    -   `process_text()`: Main function that orchestrates Step 1
-   **Dependencies**: spaCy (`en_core_web_sm`)

#### `backend/step2_event_extraction.py`

-   **Purpose**: Implements Step 2 - Event & Role Extraction
-   **What it does**:
    -   Extracts event frames from sentences
    -   Uses SRL (Semantic Role Labeling) when available
    -   Falls back to dependency parsing
    -   Identifies actors, actions, targets, locations, times
-   **Key Functions**:
    -   `extract_events_with_srl()`: Attempts SRL-based extraction
    -   `fill_gaps_with_dependencies()`: Uses spaCy dependencies to complete events
    -   `build_event_frames()`: Constructs complete event frames
    -   `extract_events()`: Main function that orchestrates Step 2
-   **Dependencies**: HuggingFace Transformers, spaCy

#### `backend/step3_temporal_normalization.py`

-   **Purpose**: Implements Step 3 - Temporal Normalization
-   **What it does**:
    -   Extracts time expressions from events
    -   Normalizes relative times to absolute dates
    -   Standardizes time formats
    -   Handles vague time expressions
-   **Key Functions**:
    -   `extract_time_expressions()`: Collects all time expressions
    -   `normalize_with_heideltime()`: Uses HeidelTime for normalization
    -   `normalize_time_fallback()`: Fallback regex-based normalization
    -   `attach_normalized_times()`: Updates events with normalized times
    -   `normalize_temporal_expressions()`: Main function that orchestrates Step 3
-   **Dependencies**: HeidelTime (optional, has fallback)

#### `backend/step4_semantic_representation.py`

-   **Purpose**: Implements Step 4 - Semantic Representation
-   **What it does**:
    -   Formats events as semantic strings
    -   Generates embeddings using sentence transformers
    -   Creates semantic memory table
-   **Key Functions**:
    -   `format_event_string()`: Creates human-readable event representation
    -   `generate_embeddings()`: Converts event strings to vectors
    -   `build_semantic_memory()`: Creates semantic memory DataFrame
    -   `create_semantic_representations()`: Main function that orchestrates Step 4
-   **Dependencies**: sentence-transformers, pandas, numpy

#### `backend/step5_memory_storage.py`

-   **Purpose**: Implements Step 5 - Memory Storage
-   **What it does**:
    -   Combines all processed data
    -   Extracts unique characters and entities
    -   Creates unified memory module
-   **Key Functions**:
    -   `extract_characters_entities()`: Identifies and categorizes entities
    -   `build_memory_module()`: Combines all data into unified structure
    -   `save_memory_module()`: Saves final JSON output
    -   `create_memory_module()`: Main function that orchestrates Step 5
-   **Dependencies**: Standard library (json)

#### `backend/model_cache.py`

-   **Purpose**: Model management and status checking utilities
-   **What it does**:
    -   Checks which models are installed/cached
    -   Reports model storage locations (local vs system cache)
    -   Provides status information
-   **Key Functions**:
    -   `check_spacy_model()`: Verifies spaCy model installation
    -   `check_sentence_transformer_model()`: Checks sentence transformer cache
    -   `check_huggingface_model()`: Checks HuggingFace model cache
    -   `get_model_status()`: Returns status of all models
    -   `print_model_status()`: Prints formatted status report
-   **Usage**: Run via `python -m backend.model_cache`

#### `backend/utils.py`

-   **Purpose**: Shared utility functions used across all steps
-   **What it does**:
    -   Provides common file I/O operations
    -   Handles directory management
    -   Provides helper functions
-   **Key Functions**:
    -   `load_json()`: Loads JSON files into Python dictionaries
    -   `save_json()`: Saves Python data structures to JSON files
    -   `ensure_directory()`: Creates directories if they don't exist
    -   `get_reference_date()`: Gets current date for temporal normalization
    -   `read_text_file()`: Reads plain text files

### Output Files Explained

#### `output/preprocessed/chapters.json`

-   **Generated by**: Step 1
-   **Purpose**: Stores chapter segmentation results
-   **Contains**:
    -   List of all chapters with IDs, titles, and full text
    -   Total chapter count
-   **Use case**: Chapter-level queries, chapter navigation

#### `output/preprocessed/sentences.json`

-   **Generated by**: Step 1
-   **Purpose**: Stores tokenized and annotated sentences
-   **Contains**:
    -   All sentences with full linguistic annotations
    -   Tokens, POS tags, dependencies, entities for each sentence
    -   Links to source chapters
-   **Use case**: Sentence-level analysis, linguistic queries

#### `output/memory/events.json`

-   **Generated by**: Step 2 (updated by Step 3)
-   **Purpose**: Stores all extracted event frames
-   **Contains**:
    -   Complete event frames with actors, actions, targets, locations, times
    -   Normalized timestamps (added in Step 3)
    -   Semantic roles and dependencies
-   **Use case**: Event queries, timeline analysis, rule checking

#### `output/memory/timestamps.json`

-   **Generated by**: Step 3
-   **Purpose**: Stores normalized temporal expressions
-   **Contains**:
    -   Reference date used for calculations
    -   Mapping of original time expressions to normalized forms
    -   Time types and confidence scores
-   **Use case**: Temporal analysis, timeline construction

#### `output/memory/event_embeddings.json`

-   **Generated by**: Step 4
-   **Purpose**: Stores embedding vectors for all events
-   **Contains**:
    -   List of event IDs
    -   Array of embedding vectors (384 numbers each)
    -   Model information
-   **Size**: Can be very large (millions of numbers)
-   **Use case**: Semantic similarity search, event clustering

#### `output/memory/memory_semantic.json`

-   **Generated by**: Step 4
-   **Purpose**: Stores semantic memory table with event metadata and embeddings
-   **Contains**:
    -   Event records with semantic strings
    -   Embedding vectors embedded in each record
    -   Timestamp information
-   **Use case**: Semantic queries, similarity analysis

#### `output/memory/memory_module.json` ⭐

-   **Generated by**: Step 5
-   **Purpose**: **Main output file - unified memory module**
-   **Contains**: Everything! All chapters, sentences, events, timestamps, embeddings metadata, characters, entities
-   **Use case**:
    -   Primary input for rule checking module (3.2.4)
    -   Complete story analysis
    -   Timeline verification
    -   Consistency checking
    -   Any query or analysis operation
-   **This is the 50% prototype deliverable!**

### Root Directory Files

#### `download_models.py`

-   **Purpose**: Downloads all required models to local `models/` directory
-   **What it does**:
    -   Downloads Sentence Transformer model to `models/sentence_transformers/`
    -   Downloads HuggingFace SRL model to `models/huggingface/`
    -   Installs spaCy model (to system location)
    -   Verifies downloads and reports status
-   **Usage**: `python download_models.py`
-   **Requirements**: `huggingface_hub` package

#### `requirements.txt`

-   **Purpose**: Lists all Python package dependencies
-   **Contains**: Package names and version requirements
-   **Usage**: `pip install -r requirements.txt`

#### `README.md`

-   **Purpose**: Quick reference guide
-   **Contains**: Installation, basic usage, project overview
-   **Audience**: Developers who need quick information

#### `GUIDE.md` (This File)

-   **Purpose**: Comprehensive documentation
-   **Contains**: Detailed explanations, examples, troubleshooting
-   **Audience**: Anyone who needs to understand the system deeply

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Python 3.8 or higher** installed on your system
2. **pip** package manager (comes with Python)

### Installation

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Download all models locally (RECOMMENDED):**

    To avoid downloading models during runtime, download them all upfront:

    ```bash
    python download_models.py
    ```

    This script will:

    - Download Sentence Transformer model (`all-MiniLM-L6-v2`, ~80MB) to `models/sentence_transformers/`
    - Download HuggingFace SRL model (`dbmdz/bert-large-cased-finetuned-conll03-english`, ~500MB) to `models/huggingface/`
    - Install spaCy English model (`en_core_web_sm`, ~15MB) to spaCy's system location

    ⚠️ **Note:**

    - Total download size is ~600MB
    - This may take 10-30 minutes depending on your internet connection
    - Requires `huggingface_hub` package: `pip install huggingface_hub`

    ✅ **Models are stored locally in the `models/` directory** - you can commit this to version control or share it with your team!

    **Prerequisites:** Make sure `huggingface_hub` is installed:

    ```bash
    pip install huggingface_hub
    ```

3. **Verify models are installed:**

    ```bash
    python -m backend.model_cache
    ```

    This will show which models are cached and ready to use.

### Running the System

#### Option 1: Command Line (Easiest)

**Important:** Always use `python -m backend.pipeline` (not `python backend/pipeline.py`) to avoid import errors.

```bash
python -m backend.pipeline data/your_story.txt
```

This will:

-   Process your story file
-   Create all output files in the `output/` directory
-   Generate the final `memory_module.json`

⚠️ **Common Error:** If you see `ImportError: attempted relative import with no known parent package`, you're running the file directly. Always use `python -m backend.pipeline` instead.

#### Option 2: With Custom Options

```bash
python -m backend.pipeline data/story.txt output 2025-01-15 all-MiniLM-L6-v2
```

**Parameters:**

-   `data/story.txt` - Your input file (required)
-   `output` - Where to save results (optional, default: "output")
-   `2025-01-15` - Reference date for time normalization in YYYY-MM-DD format (optional, defaults to current date)
-   `all-MiniLM-L6-v2` - Embedding model name (optional, default: "all-MiniLM-L6-v2")

**Note:** The embedding model name should match a model in `models/sentence_transformers/` or it will be downloaded from HuggingFace.

#### Option 3: Python Script

```python
from backend.pipeline import run_pipeline

# Process a story
memory_path = run_pipeline(
    input_file="data/my_story.txt",
    output_dir="output",
    reference_date="2025-01-15"
)

print(f"Memory module created at: {memory_path}")
```

---

## 📝 Input File Format

### Required Format

Your input file should be a **plain text file** (`.txt`) with chapter markers.

### Chapter Markers

The system recognizes various chapter formats:

✅ **Supported formats:**

-   `Chapter 1`
-   `CHAPTER 2`
-   `Chapter One`
-   `Chapter I`
-   `CHAP. 3`

### Example Input File

```
Chapter 1

John walked to the river yesterday. He saw a beautiful sunset in the afternoon.
The water was calm and peaceful. Mary joined him later that evening.

Chapter 2

Two days later, John visited the old library. He found an ancient book about
time travel. The librarian, Sarah, helped him understand the complex theories.
```

### Tips for Best Results

1. **Clear chapter markers**: Use consistent formatting
2. **One sentence per line** (optional, but helpful)
3. **Proper punctuation**: Helps with sentence detection
4. **Named entities**: Capitalize proper nouns (John, Mary, London)

---

## 📊 Understanding the Output

### Output Directory Structure

```
output/
├── preprocessed/
│   ├── chapters.json          # Step 1 output: Chapter data
│   └── sentences.json         # Step 1 output: Annotated sentences
└── memory/
    ├── events.json            # Step 2 output: Event frames
    ├── timestamps.json        # Step 3 output: Normalized times
    ├── event_embeddings.json  # Step 4 output: Embedding vectors
    ├── memory_semantic.json   # Step 4 output: Semantic memory table
    └── memory_module.json     # Step 5 output: MAIN OUTPUT ⭐
```

### The Main Output: `memory_module.json`

This is the file you'll use for:

-   Rule checking
-   Timeline analysis
-   Consistency verification
-   Story querying

**Key Sections:**

1. **metadata**: Summary statistics

    ```json
    {
        "total_chapters": 5,
        "total_sentences": 150,
        "total_events": 200,
        "total_characters": 10
    }
    ```

2. **chapters**: All chapter information
3. **sentences**: All sentence data with annotations
4. **events**: All extracted event frames (most important!)
5. **characters**: List of all unique characters
6. **entities**: Locations, organizations, dates, etc.

### Event Frame Example

```json
{
    "event_id": "event_1",
    "chapter_id": "chapter_1",
    "sentence_id": "sentence_1",
    "actor": "John",
    "action": "walked",
    "target": "river",
    "location": "river",
    "time_raw": "yesterday",
    "time_normalized": "2025-01-14",
    "time_type": "DATE",
    "entities": ["John", "river"],
    "roles": {
        "ARG0": "John",
        "ARG1": "river",
        "ARGM-LOC": "river",
        "ARGM-TMP": "yesterday"
    }
}
```

---

## 🔍 Use Cases & Examples

### Use Case 1: Timeline Analysis

**Question:** "When did John visit the library?"

**How to find:**

1. Load `memory_module.json`
2. Search events where `actor == "John"` and `action == "visited"` and `target == "library"`
3. Check `time_normalized` field

**Result:** `"2025-01-17"` (normalized from "two days later")

### Use Case 2: Character Tracking

**Question:** "What did Mary do in the story?"

**How to find:**

1. Load `memory_module.json`
2. Filter events where `actor == "Mary"`
3. List all actions

**Result:** List of all events where Mary is the actor

### Use Case 3: Consistency Checking

**Question:** "Are there any timeline contradictions?"

**How to check:**

1. Extract all events with normalized times
2. Sort chronologically
3. Check for logical inconsistencies (e.g., John is in two places at once)

### Use Case 4: Semantic Similarity

**Question:** "Find events similar to 'John walking to a location'"

**How to find:**

1. Use the embedding vectors from `event_embeddings.json`
2. Calculate cosine similarity
3. Return top-k similar events

---

## 🛠️ Troubleshooting

### Problem: "spaCy English model not found"

**Solution:**

```bash
python -m spacy download en_core_web_sm
```

Or run the model download script:

```bash
python download_models.py
```

### Problem: Models downloading during runtime (slow first run)

**Solution:**
Pre-download all models before running the pipeline:

```bash
python download_models.py
```

This will cache all models locally, making subsequent runs much faster.

### Problem: Want to check which models are installed

**Solution:**

```bash
python -m backend.model_cache
```

This shows the status of all required models.

### Problem: "ModuleNotFoundError: No module named 'spacy'" or other missing modules

**Solution:**

Install all dependencies:

```bash
pip install -r requirements.txt
pip install huggingface_hub  # Required for model downloads
```

### Problem: "ImportError: attempted relative import with no known parent package"

**Solution:**

You're trying to run the pipeline file directly. Always use the module syntax:

❌ **Wrong:**

```bash
python backend/pipeline.py
```

✅ **Correct:**

```bash
python -m backend.pipeline data/story.txt
```

The `-m` flag tells Python to run the module as a script, which properly handles relative imports.

### Problem: "Could not load SRL model"

**What it means:** The system will fall back to dependency parsing, which still works but may be less accurate.

**Solution:** This is a warning, not an error. The system will continue to work.

### Problem: "HeidelTime normalization failed"

**What it means:** The system will use fallback time normalization (regex-based).

**Solution:** This is a warning, not an error. Most common time expressions are still handled.

### Problem: No events extracted

**Possible causes:**

1. Input file has no verbs (actions)
2. Chapter markers not recognized
3. File encoding issues

**Solution:**

-   Check your input file format
-   Ensure chapter markers are present
-   Verify file is UTF-8 encoded

### Problem: Slow processing

**Causes:**

-   Large input files
-   First-time model downloads (if models not pre-downloaded)
-   CPU-intensive embedding generation

**Solutions:**

-   **Pre-download all models** before running the pipeline:
    ```bash
    python download_models.py
    ```
    This eliminates download delays during processing. Models are stored in `models/` directory.
-   Be patient for large files (100+ pages)
-   Models stored in `models/` directory are used automatically - no need to re-download
-   Consider using smaller embedding model (`all-MiniLM-L6-v2` is already optimized for speed)

### Problem: Want to verify models are cached

**Solution:**
Check model status:

```bash
python -m backend.model_cache
```

This shows which models are installed and where they're cached.

---

## 📖 Technical Details (For Developers)

### Architecture

The system is modular, with each step as a separate Python module:

```
backend/
├── pipeline.py              # Main orchestrator
├── step1_text_processing.py
├── step2_event_extraction.py
├── step3_temporal_normalization.py
├── step4_semantic_representation.py
├── step5_memory_storage.py
├── model_cache.py           # Model cache utilities
└── utils.py                # Shared utilities
```

### Model Storage

Models are downloaded directly to the project's `models/` directory:

-   **Sentence Transformers**: Stored in `models/sentence_transformers/` (contains `all-MiniLM-L6-v2`)
-   **HuggingFace models**: Stored in `models/huggingface/` (contains `dbmdz/bert-large-cased-finetuned-conll03-english`)
-   **spaCy models**: Installed to spaCy's system location (`~/.local/share/spacy/`) - spaCy manages its own installation

**Pre-downloading models:**
Run `python download_models.py` to download all models. This script:

-   Uses `huggingface_hub` to download models directly to `models/` directory
-   Reduces runtime delays (no downloads during pipeline execution)
-   Stores models in the project directory (easy to share/version control)
-   Ensures models are available offline
-   Speeds up first-time execution

**Requirements:** The download script requires `huggingface_hub`:

```bash
pip install huggingface_hub
```

**Note:** The `models/` directory can be large (~600MB). Consider adding it to `.gitignore` if you don't want to commit it to version control, or commit it if you want to share models with your team.

**Checking model status:**

```python
from backend.model_cache import get_model_status, print_model_status

# Check which models are installed
status = get_model_status()
print_model_status()  # Prints formatted status
```

### Dependencies

-   **spaCy**: Natural language processing
-   **transformers**: HuggingFace models for SRL
-   **sentence-transformers**: Embedding generation
-   **pandas/numpy**: Data manipulation
-   **python-heideltime**: Temporal normalization
-   **huggingface_hub**: Required for downloading models to local directory

### Performance

-   **Processing speed**: ~100-500 sentences per minute (depends on hardware)
-   **Memory usage**: ~2-4GB RAM for typical stories
-   **Output size**: ~1-10MB per story (depends on length)

### Extending the System

Each step can be run independently:

```python
from backend.step1_text_processing import process_text
from backend.step2_event_extraction import extract_events
# ... etc
```

This allows for:

-   Custom processing pipelines
-   Step-by-step debugging
-   Integration with other systems

---

## 📄 Output Files: Complete Reference

This section provides detailed explanations of each output file, their structure, and how to use them.

### Preprocessed Files (Step 1 Outputs)

#### `output/preprocessed/chapters.json`

**Purpose**: Stores chapter segmentation results from Step 1.

**Structure**:

```json
{
    "chapters": [
        {
            "chapter_id": "chapter_1",
            "chapter_number": "1",
            "title": "Chapter 1",
            "text": "Full chapter text content here..."
        }
    ],
    "total_chapters": 15,
    "total_sentences": 3540
}
```

**Fields Explained**:

-   `chapters`: Array of chapter objects
    -   `chapter_id`: Unique identifier (chapter_1, chapter_2, etc.)
    -   `chapter_number`: Chapter number as extracted from text
    -   `title`: Chapter title/marker (e.g., "Chapter 1")
    -   `text`: Complete text content of the chapter
-   `total_chapters`: Total number of chapters found
-   `total_sentences`: Total number of sentences across all chapters

**Use Cases**:

-   Chapter-level navigation
-   Chapter-based queries
-   Understanding story structure

---

#### `output/preprocessed/sentences.json`

**Purpose**: Stores tokenized and linguistically annotated sentences from Step 1.

**Structure**:

```json
{
  "sentences": [
    {
      "sentence_id": "sentence_1",
      "chapter_id": "chapter_1",
      "text": "John walked to the river yesterday.",
      "tokens": [
        {
          "text": "John",
          "lemma": "John",
          "pos": "PROPN",
          "tag": "NNP",
          "dep": "nsubj",
          "head": "walked",
          "head_pos": "VERB",
          "is_punct": false,
          "is_space": false
        }
        // ... more tokens
      ],
      "pos_tags": [...],
      "dependencies": [...],
      "entities": [
        {
          "text": "John",
          "label": "PERSON",
          "start": 0,
          "end": 4
        }
      ]
    }
  ],
  "total_sentences": 3540
}
```

**Fields Explained**:

-   `sentences`: Array of sentence objects
    -   `sentence_id`: Unique identifier (sentence_1, sentence_2, etc.)
    -   `chapter_id`: Links to source chapter
    -   `text`: Original sentence text
    -   `tokens`: Array of word tokens with full linguistic information
        -   `text`: Original word
        -   `lemma`: Base form (walked → walk)
        -   `pos`: Part of speech (NOUN, VERB, ADJ, etc.)
        -   `tag`: Detailed tag (NNP, VBD, etc.)
        -   `dep`: Dependency relation (nsubj, dobj, etc.)
        -   `head`: The word this token depends on
        -   `head_pos`: Part of speech of the head word
    -   `pos_tags`: Simplified POS tag array
    -   `dependencies`: Dependency parse relationships
    -   `entities`: Named entities found in the sentence
        -   `text`: Entity text
        -   `label`: Entity type (PERSON, LOCATION, DATE, etc.)
        -   `start`/`end`: Character positions in sentence
-   `total_sentences`: Total number of sentences processed

**Use Cases**:

-   Sentence-level linguistic analysis
-   Finding sentences containing specific entities
-   Understanding grammatical structure
-   Debugging extraction issues

---

### Memory Files (Steps 2-5 Outputs)

#### `output/memory/events.json`

**Purpose**: Stores all extracted event frames with complete semantic role information.

**Structure**:

```json
{
  "events": [
    {
      "event_id": "event_1",
      "chapter_id": "chapter_1",
      "sentence_id": "sentence_1",
      "action": "walked",
      "action_lemma": "walk",
      "actor": "John",
      "target": "river",
      "location": "river",
      "time_raw": "yesterday",
      "time_normalized": "2025-01-14",
      "time_type": "DATE",
      "time_confidence": 0.8,
      "entities": ["John", "river"],
      "roles": {
        "ARG0": "John",
        "ARG1": "river",
        "ARGM-LOC": "river",
        "ARGM-TMP": "yesterday"
      },
      "dependencies": [...]
    }
  ],
  "total_events": 5746
}
```

**Fields Explained**:

-   `events`: Array of event frame objects
    -   `event_id`: Unique identifier (event_1, event_2, etc.)
    -   `chapter_id`/`sentence_id`: Source location
    -   `action`: The verb/predicate (what happened)
    -   `action_lemma`: Base form of the verb
    -   `actor`: Subject/ARG0 (who performed the action)
    -   `target`: Object/ARG1/ARG2 (what/who was affected)
    -   `location`: Location modifier/ARGM-LOC (where it happened)
    -   `time_raw`: Original time expression/ARGM-TMP
    -   `time_normalized`: Standardized timestamp (from Step 3)
    -   `time_type`: Type of time (DATE, TIME, RELATIVE, UNKNOWN)
    -   `time_confidence`: Confidence in time normalization (0.0-1.0)
    -   `entities`: Named entities from the sentence
    -   `roles`: Dictionary of semantic roles (ARG0, ARG1, ARGM-LOC, ARGM-TMP)
    -   `dependencies`: Full dependency parse information
-   `total_events`: Total number of events extracted

**Use Cases**:

-   **Rule checking**: Verify story rules and constraints
-   **Timeline analysis**: Sort events chronologically
-   **Character tracking**: Find all events for a specific character
-   **Event queries**: Find events matching specific criteria
-   **Consistency checking**: Detect contradictions

---

#### `output/memory/timestamps.json`

**Purpose**: Stores normalized temporal expressions and normalization metadata.

**Structure**:

```json
{
    "reference_date": "2025-11-29",
    "normalized_times": {
        "yesterday": {
            "original": "yesterday",
            "normalized": "2025-11-28",
            "time_type": "DATE",
            "confidence": 0.8
        },
        "two days later": {
            "original": "two days later",
            "normalized": "2025-12-01",
            "time_type": "DATE",
            "confidence": 0.8
        }
    },
    "total_expressions": 234
}
```

**Fields Explained**:

-   `reference_date`: Date used as reference for relative time calculations (YYYY-MM-DD format)
-   `normalized_times`: Dictionary mapping original expressions to normalized data
    -   Key: Original time expression
    -   Value: Normalization result object
        -   `original`: The original time expression
        -   `normalized`: Standardized form
        -   `time_type`: Type (DATE, TIME, RELATIVE, UNKNOWN)
        -   `confidence`: Confidence score (0.0-1.0)
-   `total_expressions`: Number of unique time expressions found

**Use Cases**:

-   Understanding time normalization results
-   Debugging temporal issues
-   Timeline construction
-   Temporal consistency verification

---

#### `output/memory/event_embeddings.json`

**Purpose**: Stores embedding vectors for semantic similarity search.

**Structure**:

```json
{
  "event_ids": ["event_1", "event_2", "event_3", ...],
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // 384 numbers for event_1
    [0.234, -0.567, 0.890, ...],  // 384 numbers for event_2
    [0.345, -0.678, 0.901, ...]   // 384 numbers for event_3
  ],
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2"
}
```

**Fields Explained**:

-   `event_ids`: List of event IDs in the same order as embeddings array
-   `embeddings`: Array of arrays, where each inner array is a 384-dimensional vector
    -   Each vector represents the semantic meaning of an event
    -   Similar events have similar vectors (measured by cosine similarity)
-   `embedding_dim`: Dimension of each vector (384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2)
-   `model_name`: Name of the sentence transformer model used

**Important Notes**:

-   This file can be **very large** (millions of numbers for long stories)
-   Vectors are stored as lists of floats
-   Use NumPy for efficient vector operations
-   Full vectors are here; memory_module.json contains metadata only

**Use Cases**:

-   Semantic similarity search
-   Event clustering
-   Finding similar events
-   Pattern detection

---

#### `output/memory/memory_semantic.json`

**Purpose**: Stores semantic memory table with event metadata and embeddings combined.

**Structure**:

```json
{
  "events": [
    {
      "event_id": "event_1",
      "chapter_id": "chapter_1",
      "sentence_id": "sentence_1",
      "semantic_string": "Actor: John; Action: walked; Target: river; Time: 2025-01-14.",
      "normalized_timestamp": "2025-01-14",
      "timestamp_type": "DATE",
      "embedding_vector": [0.123, -0.456, 0.789, ...],
      "embedding_dim": 384
    }
  ],
  "total_events": 5746,
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2"
}
```

**Fields Explained**:

-   `events`: Array of event records with semantic information
    -   `event_id`, `chapter_id`, `sentence_id`: Identifiers
    -   `semantic_string`: Human-readable event representation
    -   `normalized_timestamp`: Standardized time
    -   `timestamp_type`: Time type
    -   `embedding_vector`: Full 384-dimensional embedding
    -   `embedding_dim`: Embedding dimension
-   `total_events`: Total number of events
-   `embedding_dim`: Dimension of embeddings
-   `model_name`: Model used

**Use Cases**:

-   Semantic queries with embeddings
-   Event similarity analysis
-   Combined semantic and temporal queries

---

#### `output/memory/memory_module.json` ⭐

**Purpose**: **Main unified output - the 50% prototype deliverable!**

This is the single file that contains everything needed for rule checking, timeline analysis, and story understanding.

**Complete Structure**:

```json
{
  "metadata": {
    "total_chapters": 15,
    "total_sentences": 3540,
    "total_events": 5746,
    "total_characters": 146,
    "total_locations": 20,
    "total_organizations": 26,
    "embedding_dim": 384,
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "chapters": [...],        // Complete chapter data from Step 1
  "sentences": [...],       // Complete sentence data from Step 1
  "events": [...],          // All event frames from Steps 2-3
  "timestamps": {...},      // Time normalization data from Step 3
  "embeddings": {
    "event_ids": [...],
    "embedding_dim": 384,
    "model_name": "all-MiniLM-L6-v2",
    "note": "Full embedding vectors stored in event_embeddings.json"
  },
  "characters": ["John", "Mary", "Rafe", ...],
  "entities": {
    "locations": ["river", "Riverside", "Manila", ...],
    "organizations": ["Philippine Armed Forces", ...],
    "dates": ["2025-01-14", ...],
    "times": ["T-AFTERNOON", ...],
    "other": [...]
  }
}
```

**Section Details**:

1. **`metadata`**: Summary statistics

    - Quick overview of story size
    - Useful for understanding scope
    - Includes model information

2. **`chapters`**: All chapter data (from `chapters.json`)

    - Complete chapter structure
    - Full text content
    - Chapter identifiers

3. **`sentences`**: All sentence data (from `sentences.json`)

    - Complete linguistic annotations
    - Tokens, POS, dependencies, entities
    - Links to chapters

4. **`events`**: All event frames (from `events.json`)

    - **This is the core data for analysis**
    - Complete event information
    - Normalized timestamps
    - Semantic roles

5. **`timestamps`**: Time normalization data (from `timestamps.json`)

    - Reference date
    - Normalized time mappings
    - Total expressions

6. **`embeddings`**: Embedding metadata (from `event_embeddings.json`)

    - Event ID list
    - Model information
    - Note about full vectors location

7. **`characters`**: List of all unique characters

    - Extracted from actors and PERSON entities
    - Sorted alphabetically
    - Useful for character tracking

8. **`entities`**: Categorized entities
    - `locations`: All location entities (GPE, LOC)
    - `organizations`: All organization entities (ORG)
    - `dates`: Date entities
    - `times`: Time entities
    - `other`: Other entity types

**Use Cases**:

-   **Primary input for rule checking module (3.2.4)**
-   Complete story analysis
-   Timeline verification
-   Consistency checking
-   Character relationship analysis
-   Any query or analysis operation

**File Size**: Can be large (several MB for long stories) but contains everything in one place.

---

## ❓ Frequently Asked Questions

### Q: What file formats are supported?

**A:** Currently only plain text (`.txt`) files. The system expects chapter markers in the text.

### Q: Can I process multiple stories at once?

**A:** Yes, run the pipeline multiple times with different input files. Each will create its own output directory.

### Q: How accurate is the event extraction?

**A:** Accuracy depends on:

-   Input text quality
-   Sentence structure complexity
-   Model availability

Typical accuracy: 80-90% for well-structured narratives.

### Q: Can I use this for non-English text?

**A:** Currently only English is supported. The system uses English-specific models.

### Q: What's the maximum file size?

**A:** There's no hard limit, but very large files (>100MB) may take significant time to process.

### Q: Can I customize the output format?

**A:** Yes, you can modify the step modules to change output structure. The JSON format is flexible.

### Q: How do I query the memory module?

**A:** Load the JSON file in Python and use standard dictionary/list operations:

```python
import json

with open('output/memory/memory_module.json') as f:
    memory = json.load(f)

# Find all events with John as actor
john_events = [e for e in memory['events'] if e.get('actor') == 'John']
```

---

## 📞 Getting Help

### Common Issues

1. **Check the Troubleshooting section above**
2. **Verify your input file format**
3. **Ensure all dependencies are installed**
4. **Check Python version (3.8+)**

### Debugging Tips

1. **Run steps individually** to isolate issues
2. **Check intermediate outputs** in `output/preprocessed/`
3. **Enable verbose logging** (add print statements)
4. **Test with sample story** first (`data/sample_story.txt`)

---

## 🎓 Learning More

### Key Concepts

-   **Semantic Role Labeling (SRL)**: Identifying who did what to whom
-   **Dependency Parsing**: Understanding grammatical relationships
-   **Named Entity Recognition (NER)**: Finding people, places, dates
-   **Embeddings**: Converting text to numerical vectors
-   **Temporal Normalization**: Standardizing time expressions

### Further Reading

-   spaCy documentation: https://spacy.io/
-   HuggingFace Transformers: https://huggingface.co/docs/transformers
-   Sentence Transformers: https://www.sbert.net/

---

## 📄 Summary

The Temporal Memory Layer is a powerful system that:

✅ Processes long-form narratives automatically  
✅ Extracts structured event information  
✅ Normalizes temporal expressions  
✅ Creates searchable semantic representations  
✅ Generates unified memory modules for analysis

**Main Output:** `output/memory/memory_module.json`

This file contains everything you need for:

-   Rule checking
-   Timeline analysis
-   Consistency verification
-   Story understanding
-   Query and search operations

---

_Last Updated: 2025_
