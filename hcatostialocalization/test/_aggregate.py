import os
import re
import ast
import statistics
import sys

def aggregate_log(file_path):
    # Regex to capture the 6 metric columns at the end of each line
    pattern = re.compile(r'(\S+)\s+(\[[^\]]+\])\s+(\S+)\s+(\[[^\]]+\])\s+(\S+)\s+(\[[^\]]+\])\s*$')
    
    metrics = {
        'distance': [],
        'distance_per_class': [],
        'peak_confidence': [],
        'peak_confidence_per_class': [],
        'outlier': [],
        'outlier_per_class': []
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    for line in lines:
        match = pattern.search(line.strip())
        if not match:
            continue
            
        dist, dist_c, peak, peak_c, out, out_c = match.groups()

        # Helper to parse scalar values
        def parse_scalar(val):
            return None if val == 'None' else float(val)

        # Helper to parse list strings (e.g., "[1.2, None]")
        def parse_list(val_str):
            try:
                return ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                return []

        # Extract Scalars
        for key, val in zip(['distance', 'peak_confidence', 'outlier'], [dist, peak, out]):
            parsed_val = parse_scalar(val)
            if parsed_val is not None:
                metrics[key].append(parsed_val)
        
        # Extract Lists
        for key, val_str in zip(['distance_per_class', 'peak_confidence_per_class', 'outlier_per_class'], [dist_c, peak_c, out_c]):
            parsed_list = parse_list(val_str)
            
            # Initialize empty lists for each class index if not done yet
            if not metrics[key]:
                metrics[key] = [[] for _ in range(len(parsed_list))]
                
            for i, val in enumerate(parsed_list):
                if val is not None:
                    metrics[key][i].append(float(val))

    def get_stats(data):
        count = len(data)
        if not data:
            return count, "N/A", "N/A"
        return count, statistics.mean(data), statistics.median(data)

    output = []
    output.append("Aggregated Metrics (Excluding Nones)\n")
    output.append("=" * 40 + "\n")

    # Format Scalar stats
    for key in ['distance', 'peak_confidence', 'outlier']:
        count_v, mean_v, med_v = get_stats(metrics[key])
        output.append(f"{key}:\n  Count: {count_v}\n  Mean: {mean_v}\n  Median: {med_v}\n")

    # Format List stats
    for key in ['distance_per_class', 'peak_confidence_per_class', 'outlier_per_class']:
        output.append(f"{key}:")
        for i, class_data in enumerate(metrics[key]):
            count_v, mean_v, med_v = get_stats(class_data)
            output.append(f"\n  Class {i}:\n    Count: {count_v}\n    Mean: {mean_v}\n    Median: {med_v}")
        output.append("\n\n")

    # Write output to the same directory
    output_dir = os.path.dirname(os.path.abspath(file_path))
    output_file = os.path.join(output_dir, 'aggregated_log.txt')
    
    with open(output_file, 'w') as f:
        f.writelines(output)
        
    print(f"Aggregation complete. Results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate.py /path/to/logfile.txt")
        sys.exit(1)
        
    input_file = sys.argv[1]
    aggregate_log(input_file)

