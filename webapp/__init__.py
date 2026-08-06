"""FastAPI web interface for the Voronoi genetic art algorithm.

The package wraps the existing evolution code (``voronoi_painting``,
``evolve_voronoi`` and ``evolve_tiled``) in a browser-facing app: users supply a
target image by upload or URL, tune the hyperparameters, and watch the run
progress live. The algorithm itself is untouched - everything here is
orchestration around it.
"""
