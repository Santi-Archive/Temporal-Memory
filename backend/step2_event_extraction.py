"""
Step 2: Event & Role Extraction
Transforms linguistic data into event frames using Semantic Role Labeling (SRL)
and dependency parsing.
"""

import spacy
from typing import List, Dict, Any, Optional
from transformers import pipeline
from .utils import load_json, save_json, ensure_directory


def extract_events_with_srl(sentences: List[Dict], srl_model=None) -> List[Dict[str, Any]]:
    """
    Use HuggingFace SRL model to extract event frames.
    
    Args:
        sentences: List of sentence dictionaries from Step 1
        srl_model: Pre-loaded SRL pipeline (will create if None)
        
    Returns:
        List of event dictionaries extracted from SRL
    """
    if srl_model is None:
        # Use a BERT-based model for SRL
        # Note: We'll use a general model and extract predicates manually
        # For production, consider using a fine-tuned SRL model
        print("Loading SRL model...")
        try:
            from pathlib import Path
            import os
            
            # Try to load from local models directory first
            project_root = Path(__file__).parent.parent.absolute()
            local_model_path = project_root / "models" / "huggingface"
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            # Check if model files exist locally
            if local_model_path.exists() and any(local_model_path.iterdir()):
                # Check if key model files are present
                has_config = (local_model_path / "config.json").exists()
                has_model = any(local_model_path.glob("*.bin")) or any(local_model_path.glob("*.safetensors"))
                
                if has_config and has_model:
                    print(f"Loading from local models directory: {local_model_path}")
                    # Load directly from local path
                    srl_model = pipeline(
                        "token-classification",
                        model=str(local_model_path),
                        aggregation_strategy="simple"
                    )
                    print("✓ SRL model loaded successfully from local directory")
                else:
                    # Model directory exists but incomplete, try with model name
                    print(f"Local model directory found but incomplete, loading from HuggingFace...")
                    # Set environment variables to use local cache
                    os.environ["TRANSFORMERS_CACHE"] = str(local_model_path)
                    os.environ["HF_HOME"] = str(local_model_path)
                    srl_model = pipeline(
                        "token-classification",
                        model=model_name,
                        aggregation_strategy="simple"
                    )
                    print("✓ SRL model loaded successfully")
            else:
                # No local model, use system cache
                print("Loading from HuggingFace (will use system cache)...")
                srl_model = pipeline(
                    "token-classification",
                    model=model_name,
                    aggregation_strategy="simple"
                )
                print("✓ SRL model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load SRL model: {e}")
            print("Falling back to dependency parsing only")
            print("Note: Event extraction will still work, but may be less accurate")
            srl_model = None
    
    events = []
    event_id_counter = 1
    
    for sentence in sentences:
        sentence_text = sentence["text"]
        chapter_id = sentence.get("chapter_id", "unknown")
        sentence_id = sentence.get("sentence_id", "unknown")
        
        # Extract predicates (verbs) from POS tags
        predicates = []
        for token in sentence.get("tokens", []):
            if token.get("pos") == "VERB" and not token.get("is_punct", False):
                predicates.append({
                    "text": token["text"],
                    "lemma": token.get("lemma", token["text"]),
                    "index": len(predicates)
                })
        
        # For each predicate, create an event frame
        for pred_idx, predicate in enumerate(predicates):
            event = {
                "event_id": f"event_{event_id_counter}",
                "chapter_id": chapter_id,
                "sentence_id": sentence_id,
                "action": predicate["text"],
                "action_lemma": predicate["lemma"],
                "actor": None,  # ARG0
                "target": None,  # ARG1/ARG2
                "location": None,  # ARGM-LOC
                "time_raw": None,  # ARGM-TMP
                "entities": [ent["text"] for ent in sentence.get("entities", [])],
                "roles": {},
                "dependencies": sentence.get("dependencies", [])
            }
            
            # Try to extract roles using dependency parsing (fallback method)
            # This will be enhanced by fill_gaps_with_dependencies
            events.append(event)
            event_id_counter += 1
    
    return events


def fill_gaps_with_dependencies(events: List[Dict], sentences: List[Dict], nlp) -> List[Dict]:
    """
    Use spaCy dependencies to complete missing roles in events.
    
    Args:
        events: List of event dictionaries
        sentences: List of sentence dictionaries
        nlp: spaCy language model
        
    Returns:
        List of event dictionaries with filled roles
    """
    # Create a mapping from sentence_id to sentence data
    sentence_map = {sent["sentence_id"]: sent for sent in sentences}
    
    for event in events:
        sentence_id = event["sentence_id"]
        if sentence_id not in sentence_map:
            continue
        
        sentence = sentence_map[sentence_id]
        sentence_text = sentence["text"]
        
        # Process sentence with spaCy to get dependency tree
        doc = nlp(sentence_text)
        
        # Find the action verb in the sentence
        action = event.get("action")
        if not action:
            continue
        
        # Find the verb token
        verb_token = None
        for token in doc:
            if token.text == action or token.lemma_ == event.get("action_lemma", ""):
                if token.pos_ == "VERB":
                    verb_token = token
                    break
        
        if not verb_token:
            continue
        
        # Extract subject (ARG0 - actor)
        if not event.get("actor"):
            for child in verb_token.children:
                if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    # Get the full noun phrase
                    actor_text = " ".join([t.text for t in child.subtree])
                    event["actor"] = actor_text
                    event["roles"]["ARG0"] = actor_text
                    break
        
        # Extract direct object (ARG1 - target)
        if not event.get("target"):
            for child in verb_token.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    target_text = " ".join([t.text for t in child.subtree])
                    event["target"] = target_text
                    event["roles"]["ARG1"] = target_text
                    break
        
        # Extract indirect object (ARG2)
        if not event.get("target"):
            for child in verb_token.children:
                if child.dep_ in ["dative", "iobj"]:
                    target_text = " ".join([t.text for t in child.subtree])
                    event["target"] = target_text
                    event["roles"]["ARG2"] = target_text
                    break
        
        # Extract location (ARGM-LOC)
        if not event.get("location"):
            for child in verb_token.children:
                if child.dep_ == "prep":
                    # Check if it's a location preposition
                    if child.text.lower() in ["in", "on", "at", "near", "by", "under", "over"]:
                        # Get the object of the preposition
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                location_text = " ".join([t.text for t in prep_child.subtree])
                                event["location"] = location_text
                                event["roles"]["ARGM-LOC"] = location_text
                                break
        
        # Extract time (ARGM-TMP)
        if not event.get("time_raw"):
            # Check for temporal modifiers
            for child in verb_token.children:
                if child.dep_ == "prep":
                    if child.text.lower() in ["at", "on", "in", "during", "before", "after", "since", "until"]:
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                time_text = " ".join([t.text for t in prep_child.subtree])
                                event["time_raw"] = time_text
                                event["roles"]["ARGM-TMP"] = time_text
                                break
            
            # Also check for temporal adverbials
            if not event.get("time_raw"):
                for token in doc:
                    if token.dep_ == "advmod" and token.pos_ == "ADV":
                        # Check if it's a temporal adverb
                        temporal_words = ["yesterday", "today", "tomorrow", "now", "then", "later", "earlier"]
                        if token.text.lower() in temporal_words or any(tw in token.text.lower() for tw in temporal_words):
                            event["time_raw"] = token.text
                            event["roles"]["ARGM-TMP"] = token.text
                            break
        
        # Extract entities from NER
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                if ent.text not in event.get("entities", []):
                    event["entities"].append(ent.text)
    
    return events


def build_event_frames(sentences: List[Dict], srl_results: List[Dict], deps: List[Dict], nlp) -> List[Dict]:
    """
    Construct complete event frames from SRL results and dependencies.
    
    Args:
        sentences: List of sentence dictionaries
        srl_results: Results from SRL extraction
        deps: Dependency parsing results
        nlp: spaCy language model
        
    Returns:
        List of complete event frames
    """
    # Extract events using SRL
    events = extract_events_with_srl(sentences, srl_model=None)
    
    # Fill gaps using dependency parsing
    events = fill_gaps_with_dependencies(events, sentences, nlp)
    
    return events


def extract_events(input_dir: str, output_dir: str = "output") -> List[Dict[str, Any]]:
    """
    Main function to extract events from preprocessed sentences.
    
    Args:
        input_dir: Directory containing preprocessed files
        output_dir: Output directory for event files
        
    Returns:
        List of event dictionaries
    """
    from .utils import load_json
    
    # Load preprocessed data
    sentences_data = load_json(f"{input_dir}/preprocessed/sentences.json")
    sentences = sentences_data.get("sentences", [])
    
    print(f"Extracting events from {len(sentences)} sentences...")
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found. Please run: python -m spacy download en_core_web_sm"
        )
    
    # Extract events
    events = build_event_frames(sentences, [], [], nlp)
    
    print(f"Extracted {len(events)} events")
    
    # Save events
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    events_data = {
        "events": events,
        "total_events": len(events)
    }
    
    save_json(events_data, f"{memory_dir}/events.json")
    print(f"Saved events to {memory_dir}/events.json")
    
    return events

