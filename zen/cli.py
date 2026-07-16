import sys
from zen.ingest.tensorboard import load_run
from zen.detect.rules import detect
from zen.router import route
from zen.memory.store import record

def main():
    if len(sys.argv) != 2:
        print("Usage: python cli.py <run_directory_path>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    try:
        metrics = load_run(run_dir)
        issues = detect(metrics)
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("no issues detected")
        
        tier = route(metrics, issues)
        print(f"tier: {tier}")
        
        # Record the run path, issues, and tier
        record(run_dir, issues, tier)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
