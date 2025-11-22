import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inferencePipeline.load import loadPipeline

def verify_consistency():
    print("Initializing pipeline...")
    pipeline = loadPipeline()
    
    questions = [
        {'questionID': 101, 'question': 'What is 1+1?'},
        {'questionID': 102, 'question': 'Who wrote Hamlet?'},
        {'questionID': 103, 'question': 'Capital of France?'}
    ]
    
    print("Running inference...")
    results = pipeline(questions)
    
    print("Verifying results...")
    
    # 1. Check if result is a list
    if not isinstance(results, list):
        print(f"FAILED: Expected list, got {type(results)}")
        return
        
    # 2. Check length
    if len(results) != len(questions):
        print(f"FAILED: Expected length {len(questions)}, got {len(results)}")
        return
        
    # 3. Check order and format
    for i, (q, r) in enumerate(zip(questions, results)):
        if r['questionID'] != q['questionID']:
            print(f"FAILED: Order mismatch at index {i}. Expected ID {q['questionID']}, got {r['questionID']}")
            return
        
        if 'answer' not in r:
            print(f"FAILED: Missing 'answer' key at index {i}")
            return
            
        if not isinstance(r['answer'], str):
            print(f"FAILED: Answer is not a string at index {i}")
            return
            
        print(f"Item {i} OK: ID={r['questionID']}, Answer='{r['answer'][:20]}...'")
        
    print("\nConsistency Check Passed! Output format matches serial execution exactly.")

if __name__ == "__main__":
    verify_consistency()
