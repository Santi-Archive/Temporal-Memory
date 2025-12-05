"""
Step 5: Temporal Memory Storage Layer
Constructs the final, unified memory module combining all processed data.
"""

from typing import List, Dict, Any, Set
from pathlib import Path
from .utils import load_json, save_json, ensure_directory


def extract_characters_entities(events: List[Dict], sentences: List[Dict]) -> Dict[str, Any]:
    """
    Extract unique characters and entities from events and sentences.
    
    Args:
        events: List of event dictionaries
        sentences: List of sentence dictionaries
        
    Returns:
        Dictionary containing characters and entities
    """
    characters: Set[str] = set()
    locations: Set[str] = set()
    organizations: Set[str] = set()
    dates: Set[str] = set()
    times: Set[str] = set()
    other_entities: Set[str] = set()
    
    # Extract from events
    for event in events:
        # Extract actor (likely a character)
        actor = event.get("actor")
        if actor:
            characters.add(actor.strip())
        
        # Extract entities from event
        entities = event.get("entities", [])
        for entity in entities:
            if isinstance(entity, str):
                # Simple string entity
                other_entities.add(entity.strip())
            elif isinstance(entity, dict):
                # Structured entity with label
                entity_text = entity.get("text", "").strip()
                entity_label = entity.get("label", "").upper()
                
                if entity_label == "PERSON":
                    characters.add(entity_text)
                elif entity_label in ["GPE", "LOC"]:
                    locations.add(entity_text)
                elif entity_label == "ORG":
                    organizations.add(entity_text)
                elif entity_label == "DATE":
                    dates.add(entity_text)
                elif entity_label == "TIME":
                    times.add(entity_text)
                else:
                    other_entities.add(entity_text)
    
    # Extract from sentences
    for sentence in sentences:
        entities = sentence.get("entities", [])
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity.get("text", "").strip()
                entity_label = entity.get("label", "").upper()
                
                if entity_label == "PERSON":
                    characters.add(entity_text)
                elif entity_label in ["GPE", "LOC"]:
                    locations.add(entity_text)
                elif entity_label == "ORG":
                    organizations.add(entity_text)
                elif entity_label == "DATE":
                    dates.add(entity_text)
                elif entity_label == "TIME":
                    times.add(entity_text)
                else:
                    other_entities.add(entity_text)
    
    return {
        "characters": sorted(list(characters)),
        "locations": sorted(list(locations)),
        "organizations": sorted(list(organizations)),
        "dates": sorted(list(dates)),
        "times": sorted(list(times)),
        "other_entities": sorted(list(other_entities)),
        "total_characters": len(characters),
        "total_locations": len(locations),
        "total_organizations": len(organizations)
    }


def build_memory_module(
    chapters: List[Dict],
    sentences: List[Dict],
    events: List[Dict],
    timestamps: Dict[str, Any],
    embeddings: Dict[str, Any],
    entities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine all data into unified memory module.
    
    Args:
        chapters: List of chapter dictionaries
        sentences: List of sentence dictionaries
        events: List of event dictionaries
        timestamps: Dictionary containing timestamp normalization data
        embeddings: Dictionary containing embedding data
        entities: Dictionary containing extracted entities
        
    Returns:
        Unified memory module dictionary
    """
    memory_module = {
        "metadata": {
            "total_chapters": len(chapters),
            "total_sentences": len(sentences),
            "total_events": len(events),
            "total_characters": entities.get("total_characters", 0),
            "total_locations": entities.get("total_locations", 0),
            "total_organizations": entities.get("total_organizations", 0),
            "embedding_dim": embeddings.get("embedding_dim", 0),
            "embedding_model": embeddings.get("model_name", "unknown")
        },
        "chapters": chapters,
        "sentences": sentences,
        "events": events,
        "timestamps": timestamps,
        "embeddings": {
            "event_ids": embeddings.get("event_ids", []),
            "embedding_dim": embeddings.get("embedding_dim", 0),
            "model_name": embeddings.get("model_name", "unknown"),
            "note": "Full embedding vectors stored in event_embeddings.json"
        },
        "characters": entities.get("characters", []),
        "entities": {
            "locations": entities.get("locations", []),
            "organizations": entities.get("organizations", []),
            "dates": entities.get("dates", []),
            "times": entities.get("times", []),
            "other": entities.get("other_entities", [])
        }
    }
    
    return memory_module


def save_memory_module(memory_module: Dict[str, Any], output_path: str) -> None:
    """
    Save final JSON memory module.
    
    Args:
        memory_module: Unified memory module dictionary
        output_path: Path where to save the memory module
    """
    ensure_directory(output_path.rsplit('/', 1)[0] if '/' in output_path else output_path.rsplit('\\', 1)[0])
    save_json(memory_module, output_path)
    print(f"Saved unified memory module to {output_path}")


def create_memory_module(input_dir: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Main function to create the unified memory module.
    
    Args:
        input_dir: Directory containing all processed files
        output_dir: Output directory for memory module
        
    Returns:
        Unified memory module dictionary
    """
    print("Building unified memory module...")
    
    # Load all processed data
    print("Loading preprocessed data...")
    
    # Load chapters from individual files
    chapters_dir = Path(f"{input_dir}/preprocessed/chapters")
    chapters = []
    
    if chapters_dir.exists():
        # Load all chapter JSON files
        chapter_files = sorted(chapters_dir.glob("chapter_*.json"))
        print(f"Loading {len(chapter_files)} chapter files...")
        for chapter_file in chapter_files:
            chapter_data = load_json(str(chapter_file))
            chapters.append(chapter_data)
    else:
        # Fallback: try to load from old chapters.json format
        print("Warning: chapters directory not found, trying old format...")
        try:
            chapters_data = load_json(f"{input_dir}/preprocessed/chapters.json")
            chapters = chapters_data.get("chapters", [])
        except FileNotFoundError:
            print("Error: No chapters found in either format")
            chapters = []
    
    sentences_data = load_json(f"{input_dir}/preprocessed/sentences.json")
    sentences = sentences_data.get("sentences", [])
    
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    timestamps_data = load_json(f"{input_dir}/memory/timestamps.json")
    
    embeddings_data = load_json(f"{input_dir}/memory/event_embeddings.json")
    
    # Extract characters and entities
    print("Extracting characters and entities...")
    entities = extract_characters_entities(events, sentences)
    
    # Build unified memory module
    print("Combining all data into memory module...")
    memory_module = build_memory_module(
        chapters=chapters,
        sentences=sentences,
        events=events,
        timestamps=timestamps_data,
        embeddings=embeddings_data,
        entities=entities
    )
    
    # Save memory module
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_memory_module(memory_module, f"{memory_dir}/memory_module.json")
    
    print(f"Memory module created successfully!")
    print(f"  - {len(chapters)} chapters")
    print(f"  - {len(sentences)} sentences")
    print(f"  - {len(events)} events")
    print(f"  - {entities.get('total_characters', 0)} characters")
    print(f"  - {entities.get('total_locations', 0)} locations")
    
    return memory_module

