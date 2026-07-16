import sys
from zen.ingest.tensorboard import load_run
from zen.detect.rules import detect
from zen.router import route
from zen.memory.store import record, history
from zen.vlak import recommend, rank, group

def main():
    if len(sys.argv) != 2:
        print("Usage: python cli.py <run_directory_path> or python cli.py history")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    try:
        if arg == "history":
            for row in history():
                print(row)
        else:
            run_dir = arg
            metrics = load_run(run_dir)
            issues = detect(metrics)
            
            if issues:
                grouped = group(rank(issues))
                modes = [issue['mode'] for issue in grouped]
                for mode in modes:
                    print(mode)
                
                # Record the run path, issues, and tier
                tier = route(metrics, issues)
                record(run_dir, modes, tier)
                
                # Get recommendations based on issues
                recs = recommend(grouped)
                for rec in recs:
                    print(rec)
            else:
                print("No issues detected")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
