"""
Step 4: Semantic Representation & Memory Structuring
Converts symbolic events into meaningful numerical embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from .utils import load_json, save_json, ensure_directory


def format_event_string(event: Dict[str, Any]) -> str:
    """
    Create semantic string representation of an event.
    
    Args:
        event: Event dictionary
        
    Returns:
        Formatted semantic string
    """
    parts = []
    
    # Actor
    actor = event.get("actor")
    if actor:
        parts.append(f"Actor: {actor}")
    
    # Action
    action = event.get("action")
    if action:
        parts.append(f"Action: {action}")
    
    # Target
    target = event.get("target")
    if target:
        parts.append(f"Target: {target}")
    
    # Location
    location = event.get("location")
    if location:
        parts.append(f"Location: {location}")
    
    # Time
    time_normalized = event.get("time_normalized")
    if time_normalized:
        parts.append(f"Time: {time_normalized}")
    else:
        time_raw = event.get("time_raw")
        if time_raw:
            parts.append(f"Time: {time_raw}")
    
    # If no parts, create a minimal representation
    if not parts:
        parts.append(f"Event: {action or 'unknown'}")
    
    return "; ".join(parts) + "."


def generate_embeddings(event_strings: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Use sentence transformers to create embeddings.
    
    Args:
        event_strings: List of formatted event strings
        model_name: Name of the sentence transformer model
        
    Returns:
        NumPy array of embedding vectors
    """
    from pathlib import Path
    
    print(f"Loading sentence transformer model: {model_name}...")
    
    # Try to load from local models directory first
    project_root = Path(__file__).parent.parent.absolute()
    local_cache = project_root / "models" / "sentence_transformers"
    
    if local_cache.exists():
        print(f"Loading from local models directory: {local_cache}")
        model = SentenceTransformer(model_name, cache_folder=str(local_cache))
    else:
        print("Loading from system cache (local models not found)...")
        model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(event_strings)} events...")
    embeddings = model.encode(event_strings, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_semantic_memory(events: List[Dict], embeddings: np.ndarray) -> pd.DataFrame:
    """
    Create semantic memory table with event data and embeddings.
    
    Args:
        events: List of event dictionaries
        embeddings: NumPy array of embedding vectors
        
    Returns:
        Pandas DataFrame containing semantic memory
    """
    rows = []
    
    for idx, event in enumerate(events):
        # Format semantic string
        semantic_string = format_event_string(event)
        
        # Get embedding vector
        embedding_vector = embeddings[idx].tolist() if idx < len(embeddings) else []
        
        row = {
            "event_id": event.get("event_id", f"event_{idx}"),
            "chapter_id": event.get("chapter_id", "unknown"),
            "sentence_id": event.get("sentence_id", "unknown"),
            "semantic_string": semantic_string,
            "normalized_timestamp": event.get("time_normalized"),
            "timestamp_type": event.get("time_type"),
            "embedding_vector": embedding_vector,
            "embedding_dim": len(embedding_vector)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_semantic_representations(input_dir: str, output_dir: str = "output", model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Main function to create semantic representations and embeddings.
    
    Args:
        input_dir: Directory containing event files
        output_dir: Output directory for semantic memory files
        model_name: Name of sentence transformer model to use
        
    Returns:
        Dictionary containing semantic memory data
    """
    from .utils import load_json
    
    # Load events
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    print(f"Creating semantic representations for {len(events)} events...")
    
    # Format event strings
    print("Formatting event strings...")
    event_strings = [format_event_string(event) for event in events]
    
    # Generate embeddings
    embeddings = generate_embeddings(event_strings, model_name)
    
    # Build semantic memory DataFrame
    print("Building semantic memory table...")
    semantic_memory_df = build_semantic_memory(events, embeddings)
    
    # Convert DataFrame to dictionary for JSON serialization
    semantic_memory_dict = semantic_memory_df.to_dict(orient='records')
    
    # Prepare embedding data (store separately for efficiency)
    embeddings_data = {
        "event_ids": [event.get("event_id", f"event_{i}") for i, event in enumerate(events)],
        "embeddings": embeddings.tolist(),
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
        "model_name": model_name
    }
    
    # Prepare semantic memory data
    memory_data = {
        "events": semantic_memory_dict,
        "total_events": len(semantic_memory_dict),
        "embedding_dim": embeddings_data["embedding_dim"],
        "model_name": model_name
    }
    
    # Save outputs
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_json(embeddings_data, f"{memory_dir}/event_embeddings.json")
    print(f"Saved embeddings to {memory_dir}/event_embeddings.json")
    
    save_json(memory_data, f"{memory_dir}/memory_semantic.json")
    print(f"Saved semantic memory to {memory_dir}/memory_semantic.json")
    
    return {
        "semantic_memory": memory_data,
        "embeddings": embeddings_data
    }

