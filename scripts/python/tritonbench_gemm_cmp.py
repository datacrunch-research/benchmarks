import json
import pandas as pd
import argparse
import re

# python tritonbench_gemm_cmp.py --threshold 0.5 --file1  baseline_gemm1.jsonl --file2 baseline_gemm2.jsonl

def load_json(file_path):
    """Load JSON file and return data."""
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_benchmarks(threshold, file1, file2):
    """Compare benchmark results from two JSON files."""
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    # Extract keys common in both and containing 'avg' (the last 21 results)
    common_keys = {key for key in data1.keys() if 'avg' in key} & {key for key in data2.keys() if 'avg' in key}
    
    comparison_results = []
    for key in common_keys:

        match = re.search(r'-(tflops|latency|speedup)-', key)
        if match:
            extracted_value = match.group(1)

        value1 = data1[key]
        value2 = data2[key]
        
        # Compute differences
        changed = ((value1 - value2) / value2) * 100

        # Do we consider the change important? if true -> the change is too big (affect perf), if false -> the change is allowed (doesnt affect perf)
        valid = not (threshold > changed and changed > -threshold)

        comparison_results.append({
            'Test': key,
            'measure': extracted_value,
            'value1': value1,
            'value2': value2,
            'change_%': changed,
            'perf_dmg': valid
        })
    
    df = pd.DataFrame(comparison_results)
    return df

def main():
    parser = argparse.ArgumentParser(description='Compare two benchmark JSON results.')
    parser.add_argument('--threshold', type=float, help="threshold allowed for a value to change")
    parser.add_argument('--file1', type=str, help='Path to first JSON file')
    parser.add_argument('--file2', type=str, help='Path to second JSON file')
    parser.add_argument('--output', type=str, help='Optional output CSV file')
    
    args = parser.parse_args()
    df = compare_benchmarks(args.threshold, args.file1, args.file2)
    
    print(df)
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f'Results saved to {args.output}')

if __name__ == "__main__":
    main()