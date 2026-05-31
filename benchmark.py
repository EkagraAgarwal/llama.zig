import subprocess
import re
import csv
import sys
from datetime import datetime
import os

def run_benchmark(run_name, max_tokens=100):
    cmd = [
        r".\zig-out\bin\llama.zig.exe",
        "--model", r"models\Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "--prompt", "The quick brown fox jumps over the lazy dog and then decides to write a very long essay about the meaning of life. Here is the essay:",
        "--max-tokens", str(max_tokens)
    ]
    
    print(f"Running benchmark: {run_name} for {max_tokens} tokens...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running benchmark:\n{result.stderr}")
        return None
        
    prompt_ts = None
    gen_ts = None
    
    for line in result.stdout.split('\n'):
        # [ Prompt: 18.9 t/s | Generation: 8.7 t/s ]
        match = re.search(r'\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]', line)
        if match:
            prompt_ts = float(match.group(1))
            gen_ts = float(match.group(2))
            break
            
    if gen_ts is None:
        print("Could not parse t/s from output:")
        print(result.stdout)
        return None
        
    print(f"Result - Prompt: {prompt_ts} t/s, Generation: {gen_ts} t/s")
    
    csv_file = 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Run Name', 'Max Tokens', 'Prompt t/s', 'Generation t/s'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), run_name, max_tokens, prompt_ts, gen_ts])
        
    return gen_ts

if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "Baseline"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_benchmark(run_name, max_tokens)
