"""Hyperparameters accepted by the web interface.

The model doubles as the validation layer for the ``POST /api/jobs`` body and as
the description the front-end renders its controls from, so every field carries
its bounds and a human readable explanation.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RunParams(BaseModel):
    """Everything the user can tune for a single evolution run."""

    mode: Literal["single", "tiled"] = Field(
        default="single",
        description=(
            "Evolve the whole image as one population, or split it into "
            "independently evolved tiles that are feather-blended back together."
        ),
    )

    # --- Core evolution knobs -------------------------------------------------
    generations: int = Field(
        default=600,
        ge=1,
        le=20000,
        description="Generations to run per target (per tile in tiled mode).",
    )
    population_size: int = Field(
        default=60,
        ge=4,
        le=1000,
        description="Individuals kept in the population each generation.",
    )
    num_points: int = Field(
        default=200,
        ge=4,
        le=5000,
        description="Voronoi cells each painting starts with.",
    )
    mutation_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Fraction of points mutated in each child.",
    )
    mutation_sigma: float = Field(
        default=0.5,
        ge=0.01,
        le=5.0,
        description="Mutation strength - scales both position and colour jitter.",
    )
    survive_fraction: float = Field(
        default=0.025,
        ge=0.001,
        le=0.9,
        description="Fraction of the population that survives to breed.",
    )
    genome_duplication: bool = Field(
        default=True,
        description=(
            "Halfway through the run, merge parents so the point count doubles "
            "and the second half of the run refines a denser painting."
        ),
    )

    # --- Seeding --------------------------------------------------------------
    region_bias: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Probability a starting point is seeded inside a detected region.",
    )
    initial_colors: int = Field(
        default=60,
        ge=2,
        le=256,
        description="Colours extracted from the target before condensing.",
    )
    final_colors: int = Field(
        default=20,
        ge=1,
        le=256,
        description="Colours kept in the condensed seeding palette.",
    )

    # --- Target handling ------------------------------------------------------
    max_dimension: int = Field(
        default=400,
        ge=64,
        le=2000,
        description=(
            "Longest edge the target is scaled to before evolving. Runtime grows "
            "roughly with the pixel count, so this is the main speed dial."
        ),
    )

    # --- Tiling ---------------------------------------------------------------
    tile_rows: int = Field(default=3, ge=1, le=8, description="Tile rows.")
    tile_cols: int = Field(default=3, ge=1, le=8, description="Tile columns.")
    tile_overlap: int = Field(
        default=32,
        ge=0,
        le=256,
        description="Overlap in pixels that neighbouring tiles are blended across.",
    )

    # --- Output ---------------------------------------------------------------
    outline: bool = Field(
        default=True, description="Draw stained-glass outlines between cells."
    )
    outline_width: int = Field(default=1, ge=1, le=10, description="Outline width.")
    outline_color: str = Field(default="#000000", description="Outline colour (hex).")
    render_scale: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Supersampling factor for the final rendered image.",
    )

    # --- Execution ------------------------------------------------------------
    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description=(
            "Worker processes used to score a generation. Values above 1 only pay "
            "off for large populations, since every individual is serialized."
        ),
    )
    preview_every: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Generations between live preview refreshes.",
    )

    @field_validator("outline_color")
    @classmethod
    def _validate_hex_color(cls, value: str) -> str:
        """Normalise ``#rgb`` / ``#rrggbb`` into a lowercase ``#rrggbb`` string."""
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(character * 2 for character in text)
        if len(text) != 6 or any(c not in "0123456789abcdefABCDEF" for c in text):
            raise ValueError("outline_color must be a hex colour such as #1a1a1a")
        return f"#{text.lower()}"

    @property
    def outline_rgb(self) -> tuple:
        """The outline colour as an ``(r, g, b)`` tuple."""
        text = self.outline_color.lstrip("#")
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))

    @property
    def tile_count(self) -> int:
        """How many tiles the run will evolve (1 when not tiling)."""
        return 1 if self.mode == "single" else self.tile_rows * self.tile_cols

    @property
    def generations_per_target(self) -> int:
        """Generations spent on one target, including the duplication step."""
        return self.generations + (1 if self.genome_duplication else 0)

    @property
    def total_generations(self) -> int:
        """Total generations across the whole run, used to drive the progress bar."""
        return self.generations_per_target * self.tile_count

    @property
    def survivor_count(self) -> int:
        """Survivors per generation.

        At least two, so that ``pick_best_and_random`` can pick two distinct
        parents even for small populations and low survival fractions.
        """
        return max(2, min(self.population_size, round(self.survive_fraction * self.population_size)))

    def phase_plan(self) -> list:
        """Return the ``(label, generations)`` phases for one target.

        A run is an exploration phase, an optional one-generation genome
        duplication that doubles the point count, and a refinement phase with
        gentler mutation. This mirrors the phase structure used by the
        command-line drivers.
        """
        if not self.genome_duplication:
            return [("explore", self.generations)]

        first = max(1, self.generations // 2)
        return [
            ("explore", first),
            ("duplicate", 1),
            ("refine", self.generations - first),
        ]
