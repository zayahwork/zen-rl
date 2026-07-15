import sys
from zen.ingest.tensorboard import load_run
from zen.detect.rules import detect

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
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
