import codecs
import sys

try:
    with codecs.open('verification_output.txt', 'r', 'utf-16') as f:
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    print("--- FILTERED LOG START ---")
    for line in lines:
        l = line.strip()
        if "Inspecting" in l or "Found" in l or "Recursing" in l or "Error" in l or "FAILURE" in l or "SUCCESS" in l or "Calibrated classifier" in l:
            print(line.rstrip())
    print("--- FILTERED LOG END ---")
    
except Exception as e:
    print(f"Error reading log: {e}")
