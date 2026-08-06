"""
Benchmark harness for the stained glass genetic art algorithm.

Measures the hot paths of the evolution loop so that performance work can be
verified with numbers instead of assumptions. It is written against the public
API of ``VoronoiPainting`` only, so the exact same script can be run on any
revision of the code (e.g. ``git stash`` the optimizations and re-run to get a
baseline to compare against).

Usage:
    python benchmark.py                     # default: quick run on ./img/car.jpeg
    python benchmark.py --image ./img/girl_with_pearl_earring_half.jpg
    python benchmark.py --points 250 --population 60 --generations 5 --workers 4
    python benchmark.py --label after --save-json bench_after.json
    python benchmark.py --compare bench_before.json --save-json bench_after.json

Every timing is reported as milliseconds per operation (lower is better) except
the end-to-end figure, which is reported as generations per second (higher is
better).
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from copy import deepcopy

from PIL import Image

from voronoi_painting import VoronoiPainting

# ``evol`` ships with ``multiprocess`` (a dill-backed fork of multiprocessing),
# which is what actually serializes individuals when concurrent_workers > 1.
# Measuring with dill rather than pickle therefore reflects the real cost.
try:
    import dill as _serializer
except ImportError:  # pragma: no cover - dill comes with evol, but stay safe
    import pickle as _serializer


def timeit(fn, repeat=5, number=None, warmup=1):
    """Time ``fn`` and return (mean_ms_per_call, stdev_ms_per_call).

    :param fn: Zero-argument callable to time.
    :param repeat: Number of independent samples to take.
    :param number: Calls per sample. If None, it is auto-scaled so each sample
        takes roughly 50ms, which keeps fast operations out of timer noise.
    :param warmup: Number of untimed calls made first, so that lazily created
        caches and buffers do not pollute the first sample.
    """
    for _ in range(warmup):
        fn()

    if number is None:
        number = 1
        while True:
            start = time.perf_counter()
            for _ in range(number):
                fn()
            elapsed = time.perf_counter() - start
            if elapsed > 0.05 or number >= 10000:
                break
            number *= 4

    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(number):
                fn()
            samples.append((time.perf_counter() - start) / number * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.mean(samples), stdev


def make_painting(num_points, target_image):
    """Build a painting with a deterministic point layout for timing purposes."""
    return VoronoiPainting(num_points, target_image, background_color=(128, 128, 128))


def bench_micro(target_image, num_points, repeat):
    """Time the individual operations that the evolution loop calls the most."""
    results = {}

    painting = make_painting(num_points, target_image)
    other = make_painting(num_points, target_image)

    results["render"] = timeit(lambda: painting._render_array(), repeat=repeat)
    results["render_lines"] = timeit(
        lambda: painting._render_array_with_lines(scale=1, line_width=1), repeat=repeat
    )
    results["image_diff"] = timeit(
        lambda: painting.image_diff(target_image), repeat=repeat
    )
    results["deepcopy"] = timeit(lambda: deepcopy(painting), repeat=repeat)
    results["mutate"] = timeit(
        lambda: make_painting(num_points, target_image).mutate_points(
            rate=0.05, sigma=0.5
        ),
        repeat=repeat,
    )
    results["mutate_only"] = timeit(
        lambda: painting.mutate_points(rate=0.05, sigma=0.5), repeat=repeat
    )
    results["mate"] = timeit(
        lambda: VoronoiPainting.mate(painting, other), repeat=repeat
    )
    results["merge"] = timeit(
        lambda: VoronoiPainting.merge(painting, other), repeat=repeat
    )

    # Serialization is what the worker pool pays per individual per generation.
    payload = _serializer.dumps(painting)
    results["serialize"] = timeit(lambda: _serializer.dumps(painting), repeat=repeat)
    results["deserialize"] = timeit(lambda: _serializer.loads(payload), repeat=repeat)
    results["payload_bytes"] = (float(len(payload)), 0.0)

    return results


def bench_end_to_end(target_image, num_points, population_size, generations, workers):
    """Run a miniature version of the real evolution loop and report gens/sec."""
    from evol import Evolution, Population

    from evolve_voronoi import mate, mutate_painting, pick_best_and_random, score

    chromosomes = [
        make_painting(num_points, target_image) for _ in range(population_size)
    ]
    pop = Population(
        chromosomes=chromosomes,
        eval_function=score,
        maximize=False,
        concurrent_workers=workers,
    )

    evo = (
        Evolution()
        .survive(fraction=0.025)
        .breed(
            parent_picker=pick_best_and_random,
            combiner=mate,
            population_size=population_size,
        )
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
    )

    # One untimed generation absorbs worker start-up and first-touch caching.
    pop = pop.evolve(evo, n=1)

    start = time.perf_counter()
    pop = pop.evolve(evo, n=generations)
    elapsed = time.perf_counter() - start

    return {
        "generations_per_second": generations / elapsed,
        "seconds_per_generation": elapsed / generations,
        "seconds_total": elapsed,
        "best_fitness": float(pop.current_best.fitness),
    }


def sweep_workers(target_image, num_points, population_size, generations, worker_counts):
    """Time the same evolution at several worker counts.

    Worth running once per machine and per image size. Worker processes only pay
    off when the per-individual rendering work outweighs the cost of shipping
    individuals between processes, and that crossover moves with image size,
    point count and core count.
    """
    sweep = {}
    for workers in worker_counts:
        result = bench_end_to_end(
            target_image,
            num_points=num_points,
            population_size=population_size,
            generations=generations,
            workers=workers,
        )
        sweep[str(workers)] = result
        print(
            f"  {workers:>2} worker(s): {result['generations_per_second']:.3f} gen/s "
            f"({result['seconds_per_generation']:.3f} s/gen)"
        )
    best = max(sweep, key=lambda key: sweep[key]["generations_per_second"])
    print(f"  fastest: {best} worker(s)")
    return sweep


def format_report(results, compare=None):
    """Render the results table, optionally with a speedup column vs. a baseline."""
    lines = []
    micro = results["micro"]
    base_micro = (compare or {}).get("micro", {})

    header = f"{'operation':<16}{'ms/op':>12}{'+/-':>10}"
    if base_micro:
        header += f"{'baseline':>12}{'speedup':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, (mean, stdev) in micro.items():
        if name == "payload_bytes":
            continue
        row = f"{name:<16}{mean:>12.4f}{stdev:>10.4f}"
        if name in base_micro:
            base_mean = base_micro[name][0]
            speedup = base_mean / mean if mean else float("inf")
            row += f"{base_mean:>12.4f}{speedup:>9.2f}x"
        lines.append(row)

    payload = micro["payload_bytes"][0]
    payload_row = f"{'payload KiB':<16}{payload / 1024:>12.1f}{'':>10}"
    if "payload_bytes" in base_micro:
        base_payload = base_micro["payload_bytes"][0]
        payload_row += (
            f"{base_payload / 1024:>12.1f}"
            f"{(base_payload / payload if payload else float('inf')):>9.2f}x"
        )
    lines.append(payload_row)

    if results.get("end_to_end"):
        e2e = results["end_to_end"]
        lines.append("")
        lines.append(
            f"end-to-end: {e2e['generations_per_second']:.3f} gen/s "
            f"({e2e['seconds_per_generation']:.3f} s/gen), "
            f"best fitness {e2e['best_fitness']:.0f}"
        )
        base_e2e = (compare or {}).get("end_to_end")
        if base_e2e:
            speedup = (
                e2e["generations_per_second"] / base_e2e["generations_per_second"]
            )
            lines.append(
                f"  baseline:  {base_e2e['generations_per_second']:.3f} gen/s "
                f"-> {speedup:.2f}x faster"
            )

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="./img/car.jpeg", help="Target image path.")
    parser.add_argument(
        "--points", type=int, default=250, help="Points per painting. Default: 250."
    )
    parser.add_argument(
        "--population", type=int, default=60, help="Population for the end-to-end run."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Timed generations in the end-to-end run. 0 skips it.",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Worker processes for the end-to-end run."
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="Samples per micro-benchmark."
    )
    parser.add_argument(
        "--sweep-workers",
        help="Comma separated worker counts to time, e.g. 1,2,4,8. "
        "Use this to find the fastest setting on your machine.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument("--label", default="run", help="Label stored in the JSON dump.")
    parser.add_argument("--save-json", help="Write the raw results to this path.")
    parser.add_argument("--compare", help="Path to a previous JSON dump to compare to.")
    args = parser.parse_args(argv)

    import random

    random.seed(args.seed)

    if not os.path.exists(args.image):
        parser.error(f"target image not found: {args.image}")

    target_image = Image.open(args.image).convert("RGBA")
    print(
        f"[{args.label}] image {os.path.basename(args.image)} {target_image.size}, "
        f"{args.points} points, python {sys.version.split()[0]}\n"
    )

    results = {
        "label": args.label,
        "image": args.image,
        "size": list(target_image.size),
        "points": args.points,
        "micro": bench_micro(target_image, args.points, args.repeat),
    }

    if args.generations > 0:
        results["end_to_end"] = bench_end_to_end(
            target_image,
            num_points=args.points,
            population_size=args.population,
            generations=args.generations,
            workers=args.workers,
        )

    if args.sweep_workers:
        print("\nworker sweep")
        results["worker_sweep"] = sweep_workers(
            target_image,
            num_points=args.points,
            population_size=args.population,
            generations=args.generations or 5,
            worker_counts=[int(n) for n in args.sweep_workers.split(",") if n.strip()],
        )
        print()

    compare = None
    if args.compare:
        with open(args.compare) as handle:
            compare = json.load(handle)

    print(format_report(results, compare))

    if args.save_json:
        with open(args.save_json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved results to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
