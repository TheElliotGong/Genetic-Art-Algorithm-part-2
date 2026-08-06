"""Launch the web interface for the Voronoi genetic art algorithm.

    python run_web.py                 # http://127.0.0.1:8000
    python run_web.py --host 0.0.0.0 --port 8080

Everything heavy stays behind the ``__main__`` guard: when a run uses more than
one worker process, ``multiprocess`` re-imports this module in each child, and
an unguarded ``uvicorn.run`` would start a server per worker.
"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. Development only - the reloader and the "
        "evolution worker processes do not mix well.",
    )
    args = parser.parse_args()

    print(f"Voronoi genetic art running on http://{args.host}:{args.port}")
    uvicorn.run(
        "webapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
