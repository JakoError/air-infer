#!/usr/bin/env python3
"""
Script to transform JSON performance test results into CSV format.
Processes all JSON files in the 20260122results directory.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_result_data(json_file: Path, stage: str) -> Dict[str, Any]:
    """
    Extract and flatten data from a JSON result file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Start with stage and filename
    result = {
        'stage': stage,
        'filename': json_file.name,
    }
    
    # Add timestamp
    if 'timestamp' in data:
        result['timestamp'] = data['timestamp']
    
    # Flatten configuration
    if 'configuration' in data:
        config_flat = flatten_dict(data['configuration'], 'config')
        result.update(config_flat)
    
    # Flatten statistics (excluding raw data arrays)
    if 'statistics' in data:
        stats = data['statistics'].copy()
        # Remove raw arrays as they're too large for CSV
        stats.pop('raw_latencies_ms', None)
        stats.pop('raw_message_sizes_bytes', None)
        
        stats_flat = flatten_dict(stats, 'stats')
        result.update(stats_flat)
    
    return result


def get_all_json_files(results_dir: Path) -> List[tuple]:
    """
    Get all JSON files organized by stage.
    Returns list of (stage, json_file_path) tuples.
    """
    json_files = []
    
    for stage_dir in results_dir.iterdir():
        if stage_dir.is_dir():
            stage_name = stage_dir.name
            for json_file in stage_dir.glob('*.json'):
                json_files.append((stage_name, json_file))
    
    # Sort by stage and filename for consistent ordering
    json_files.sort(key=lambda x: (x[0], x[1].name))
    
    return json_files


def main():
    """
    Main function to process all JSON files and create CSV.
    """
    # Get the script directory and results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / '20260122results'
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Get all JSON files
    json_files = get_all_json_files(results_dir)
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    # Process all files and collect data
    all_data = []
    all_keys = set()
    
    for stage, json_file in json_files:
        try:
            data = extract_result_data(json_file, stage)
            all_data.append(data)
            all_keys.update(data.keys())
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not all_data:
        print("No data extracted from JSON files.")
        return
    
    # Sort keys for consistent column order
    # Put important columns first
    priority_keys = ['stage', 'filename', 'timestamp', 'config_num_messages', 
                     'config_message_size_bytes', 'stats_total_messages', 
                     'stats_success_count', 'stats_failure_count']
    
    sorted_keys = []
    for key in priority_keys:
        if key in all_keys:
            sorted_keys.append(key)
            all_keys.remove(key)
    
    # Add remaining keys in sorted order
    sorted_keys.extend(sorted(all_keys))
    
    # Write to CSV
    output_file = script_dir / 'results_20260122.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Successfully created CSV file: {output_file}")
    print(f"Processed {len(all_data)} result files")
    print(f"CSV contains {len(sorted_keys)} columns")


if __name__ == '__main__':
    main()

