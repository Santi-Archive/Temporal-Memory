"""
Step 3: Temporal Normalization
Converts vague or relative time phrases into standardized timestamps.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .utils import load_json, save_json, ensure_directory, get_reference_date

try:
    from heideltime import HeidelTime
except ImportError:
    HeidelTime = None
    print("Warning: python-heideltime not available. Using fallback normalization.")


def extract_time_expressions(events: List[Dict]) -> List[str]:
    """
    Collect all time expressions from events.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        List of unique time expressions
    """
    time_expressions = set()
    
    for event in events:
        # Get time from ARGM-TMP role
        time_raw = event.get("time_raw")
        if time_raw:
            time_expressions.add(time_raw.strip())
        
        # Get time from roles dictionary
        roles = event.get("roles", {})
        if "ARGM-TMP" in roles:
            time_expressions.add(roles["ARGM-TMP"].strip())
        
        # Get time from entities (DATE/TIME entities)
        entities = event.get("entities", [])
        for entity in entities:
            if isinstance(entity, dict):
                if entity.get("label") in ["DATE", "TIME"]:
                    time_expressions.add(entity.get("text", "").strip())
    
    return sorted(list(time_expressions))


def normalize_with_heideltime(time_expr: str, reference_date: str) -> Dict[str, Any]:
    """
    Use HeidelTime to normalize time expressions.
    
    Args:
        time_expr: Time expression to normalize
        reference_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Dictionary with normalized time information
    """
    if HeidelTime is None:
        # Fallback to custom normalization
        return normalize_time_fallback(time_expr, reference_date)
    
    try:
        # Initialize HeidelTime
        ht = HeidelTime()
        
        # Parse time expression
        # HeidelTime expects document text, so we'll create a simple document
        doc_text = f"Event happened {time_expr}."
        
        # Extract temporal expressions
        # Note: HeidelTime API may vary, this is a general approach
        result = ht.parse(doc_text, language='english', document_type='news')
        
        if result and len(result) > 0:
            # Extract normalized value from result
            normalized = result[0].get('value', time_expr)
            time_type = result[0].get('type', 'DATE')
            
            return {
                "original": time_expr,
                "normalized": normalized,
                "time_type": time_type,
                "confidence": 1.0
            }
    except Exception as e:
        print(f"Warning: HeidelTime normalization failed for '{time_expr}': {e}")
        return normalize_time_fallback(time_expr, reference_date)
    
    # Fallback if HeidelTime doesn't return results
    return normalize_time_fallback(time_expr, reference_date)


def normalize_time_fallback(time_expr: str, reference_date: str) -> Dict[str, Any]:
    """
    Fallback time normalization using regex and rule-based approach.
    
    Args:
        time_expr: Time expression to normalize
        reference_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Dictionary with normalized time information
    """
    time_expr_lower = time_expr.lower().strip()
    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    
    # Absolute dates
    date_patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', 'DATE'),  # YYYY-MM-DD
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'DATE'),  # MM/DD/YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', 'DATE'),  # MM-DD-YYYY
    ]
    
    for pattern, time_type in date_patterns:
        match = re.search(pattern, time_expr)
        if match:
            return {
                "original": time_expr,
                "normalized": time_expr,
                "time_type": time_type,
                "confidence": 0.9
            }
    
    # Relative times
    relative_patterns = {
        r'yesterday': lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d"),
        r'today': lambda d: d.strftime("%Y-%m-%d"),
        r'tomorrow': lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d"),
        r'(\d+)\s*days?\s*later': lambda d, m: (d + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*days?\s*ago': lambda d, m: (d - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*weeks?\s*later': lambda d, m: (d + timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*weeks?\s*ago': lambda d, m: (d - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*months?\s*later': lambda d, m: (d + timedelta(days=30*int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*months?\s*ago': lambda d, m: (d - timedelta(days=30*int(m.group(1)))).strftime("%Y-%m-%d"),
    }
    
    for pattern, func in relative_patterns.items():
        match = re.search(pattern, time_expr_lower)
        if match:
            try:
                if match.groups():
                    normalized = func(ref_date, match)
                else:
                    normalized = func(ref_date)
                return {
                    "original": time_expr,
                    "normalized": normalized,
                    "time_type": "DATE",
                    "confidence": 0.8
                }
            except Exception:
                pass
    
    # Time of day
    time_of_day = {
        'morning': 'T-MORNING',
        'afternoon': 'T-AFTERNOON',
        'evening': 'T-EVENING',
        'night': 'T-NIGHT',
        'noon': 'T-NOON',
        'midnight': 'T-MIDNIGHT'
    }
    
    for key, value in time_of_day.items():
        if key in time_expr_lower:
            return {
                "original": time_expr,
                "normalized": value,
                "time_type": "TIME",
                "confidence": 0.7
            }
    
    # Relative placeholders for vague times
    vague_patterns = {
        r'(\d+)\s*days?\s*later': 'REL-{}D',
        r'(\d+)\s*weeks?\s*later': 'REL-{}W',
        r'(\d+)\s*months?\s*later': 'REL-{}M',
    }
    
    for pattern, placeholder in vague_patterns.items():
        match = re.search(pattern, time_expr_lower)
        if match:
            num = match.group(1)
            return {
                "original": time_expr,
                "normalized": placeholder.format(num),
                "time_type": "RELATIVE",
                "confidence": 0.6
            }
    
    # Default: keep original with low confidence
    return {
        "original": time_expr,
        "normalized": time_expr,
        "time_type": "UNKNOWN",
        "confidence": 0.3
    }


def attach_normalized_times(events: List[Dict], normalized_times: Dict[str, Dict]) -> List[Dict]:
    """
    Update event frames with normalized timestamps.
    
    Args:
        events: List of event dictionaries
        normalized_times: Dictionary mapping time expressions to normalized data
        
    Returns:
        List of events with normalized time fields
    """
    for event in events:
        time_raw = event.get("time_raw")
        
        if time_raw and time_raw in normalized_times:
            normalized_data = normalized_times[time_raw]
            event["time_normalized"] = normalized_data["normalized"]
            event["time_type"] = normalized_data["time_type"]
            event["time_confidence"] = normalized_data.get("confidence", 0.5)
        else:
            # Check roles for time
            roles = event.get("roles", {})
            if "ARGM-TMP" in roles:
                time_from_role = roles["ARGM-TMP"]
                if time_from_role in normalized_times:
                    normalized_data = normalized_times[time_from_role]
                    event["time_normalized"] = normalized_data["normalized"]
                    event["time_type"] = normalized_data["time_type"]
                    event["time_confidence"] = normalized_data.get("confidence", 0.5)
                else:
                    event["time_normalized"] = None
                    event["time_type"] = None
            else:
                event["time_normalized"] = None
                event["time_type"] = None
    
    return events


def normalize_temporal_expressions(input_dir: str, output_dir: str = "output", reference_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to normalize temporal expressions in events.
    
    Args:
        input_dir: Directory containing event files
        output_dir: Output directory for timestamp files
        reference_date: Reference date for normalization (defaults to current date)
        
    Returns:
        Dictionary containing normalized timestamps
    """
    from .utils import load_json
    
    # Load events
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    print(f"Normalizing temporal expressions for {len(events)} events...")
    
    # Get reference date
    if reference_date is None:
        reference_date = get_reference_date()
    
    # Extract time expressions
    time_expressions = extract_time_expressions(events)
    print(f"Found {len(time_expressions)} unique time expressions")
    
    # Normalize each time expression
    normalized_times = {}
    for time_expr in time_expressions:
        normalized = normalize_with_heideltime(time_expr, reference_date)
        normalized_times[time_expr] = normalized
    
    # Attach normalized times to events
    events = attach_normalized_times(events, normalized_times)
    
    # Prepare timestamp data
    timestamps_data = {
        "reference_date": reference_date,
        "normalized_times": normalized_times,
        "total_expressions": len(time_expressions)
    }
    
    # Save timestamps
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_json(timestamps_data, f"{memory_dir}/timestamps.json")
    print(f"Saved timestamps to {memory_dir}/timestamps.json")
    
    # Update and save events with normalized times
    events_data["events"] = events
    save_json(events_data, f"{memory_dir}/events.json")
    print(f"Updated events with normalized times")
    
    return timestamps_data

