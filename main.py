import sys

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ui"

    if cmd == "fetch":
        from scripts.fetch_data import main as fetch
        fetch()
    elif cmd == "index":
        from scripts.build_index import main as build
        build()
    elif cmd == "setup":
        from scripts.fetch_data import main as fetch
        from scripts.build_index import main as build
        fetch()
        build()
    elif cmd == "ui":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    else:
        print("Usage: python main.py [fetch|index|setup|ui]")

if __name__ == "__main__":
    main()
