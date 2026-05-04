from random import shuffle, randint, choices, choice
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi
import numpy as np


class ColoredPoint:
    def __init__(self, img_width, img_height):
        self.coordinates = (randint(0, int(img_width)), randint(0, int(img_height)))
        self.color = (
            randint(0, 256),  # Random value for the Red channel
            randint(0, 256),  # Random value for the Green channel
            randint(0, 256),  # Random value for the Blue channel
            255,
        )  # The Alpha channel is fixed

    def __str__(self):
        return f"ColoredPoint at ({self.coordinates[0]}, {self.coordinates[1]}) of color ({self.color[0]}, {self.color[1]}, {self.color[2]})"

    def mutate(self, sigma=1.0):
        mutations = ["shift", "color"]
        weights = [50, 50]

        mutation_type = choices(mutations, weights=weights, k=1)[0]

        if mutation_type == "shift":
            self.coordinates = (
                self.coordinates[0] + int(randint(-10, 10) * sigma),
                self.coordinates[1] + int(randint(-10, 10) * sigma),
            )
        elif mutation_type == "color":
            red = self.color[0] + int(randint(-25, 25) * sigma)
            green = self.color[1] + int(randint(-25, 25) * sigma)
            blue = self.color[2] + int(randint(-25, 25) * sigma)

            self.color = (red, green, blue, 255)

            # Ensure color is within correct range
            self.color = tuple(min(max(c, 0), 255) for c in self.color)


class VoronoiPainting:
    def __init__(
        self, num_points, target_image, background_color=(0, 0, 0), output_scale=1.0
    ):
        self._img_width, self._img_height = target_image.size
        self.points = [
            ColoredPoint(self._img_width, self._img_height) for _ in range(num_points)
        ]
        self._background_color = (*background_color, 255)
        self.target_image = target_image
        self._output_scale = output_scale
        # Render cache keyed by scale -> PIL Image
        self._render_cache = {}
        # Dirty flag indicates cached renders are stale
        self._dirty = True

    @property
    def get_background_color(self):
        return self._background_color[:3]

    @property
    def get_img_width(self):
        return self._img_width

    @property
    def get_img_height(self):
        return self._img_height

    @property
    def get_output_scale(self):
        return self._output_scale

    @property
    def num_points(self):
        return len(self.points)

    def __repr__(self):
        return "VoronoiPainting with %d triangles" % self.num_points

    def mutate_points(self, rate=0.04, sigma=1.0):
        total_mutations = int(rate * self.num_points)
        random_indices = list(range(self.num_points))
        shuffle(random_indices)

        # mutate random triangles
        for i in range(total_mutations):
            index = random_indices[i]
            self.points[index].mutate(sigma=sigma)
        # Mark rendering cache stale after mutations
        self._dirty = True
        self._render_cache.clear()

    def shrink_points(self):
        """ """
        self.points.pop(randint(0, self.num_points - 1))
        # Mark rendering cache stale after structural change
        self._dirty = True
        self._render_cache.clear()

    def draw(self, scale=None) -> Image:
        if scale is None:
            scale = self._output_scale
        # Return cached render when available and not dirty
        cache_key = float(scale)
        if not getattr(self, "_dirty", True) and cache_key in self._render_cache:
            return self._render_cache[cache_key].copy()

        image = Image.new(
            "RGBA", (int(self._img_width * scale), int(self._img_height * scale))
        )
        draw = ImageDraw.Draw(image)

        if not hasattr(self, "_background_color"):
            self._background_color = (0, 0, 0, 255)

        draw.polygon(
            [
                (0, 0),
                (0, int(self._img_height * scale)),
                (int(self._img_width * scale), int(self._img_height * scale)),
                (int(self._img_width * scale), 0),
            ],
            fill=self._background_color,
        )

        vor = Voronoi([p.coordinates for p in self.points], qhull_options="Qc")

        for point, region_idx in zip(self.points, vor.point_region):
            polygon = []
            draw = True
            for vertex_idx in vor.regions[region_idx]:
                if vertex_idx == -1:
                    draw = False
                i, j = vor.vertices[vertex_idx]
                # if i < 0 or j < 0 or i > self._img_width or j > self._img_height:
                #     draw = False
                polygon.append((i * scale, j * scale))

            if draw:
                new_polygon = Image.new(
                    "RGBA",
                    (int(self._img_width * scale), int(self._img_height * scale)),
                )
                pdraw = ImageDraw.Draw(new_polygon)
                pdraw.polygon(polygon, fill=point.color)
                image = Image.alpha_composite(image, new_polygon)

        # Cache rendered result and mark clean
        try:
            self._render_cache[cache_key] = image.copy()
            self._dirty = False
        except Exception:
            # If caching fails for any reason, just return the image
            pass

        return image

    @staticmethod
    def _mate_possible(a, b) -> bool:
        return all(
            [
                a.num_points == b.num_points,
                a.get_img_width == b.get_img_width,
                a.get_img_height == b.get_img_height,
                a.get_output_scale == b.get_output_scale,
            ]
        )

    @staticmethod
    def mate(a, b):
        if not VoronoiPainting._mate_possible(a, b):
            if a.num_points > b.num_points:
                return a, a
            else:
                return b, b

        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i]) / 2) for i in range(3))

        child_a = VoronoiPainting(
            0,
            a.target_image,
            background_color=new_background,
            output_scale=a.get_output_scale,
        )
        child_b = VoronoiPainting(
            0,
            a.target_image,
            background_color=new_background,
            output_scale=a.get_output_scale,
        )

        for point_a, point_b in zip(a.points, b.points):
            if randint(0, 1) == 0:
                child_a.points.append(point_a)
                child_b.points.append(point_b)
            else:
                child_a.points.append(point_b)
                child_b.points.append(point_a)

        # Children start dirty (have new point lists)
        child_a._dirty = True
        child_b._dirty = True
        child_a._render_cache = {}
        child_b._render_cache = {}

        return child_a, child_b

    @staticmethod
    def merge(a, b):
        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i]) / 2) for i in range(3))

        merger = VoronoiPainting(
            0,
            a.target_image,
            background_color=new_background,
            output_scale=a.get_output_scale,
        )

        for point_a, point_b in zip(a.points, b.points):
            merger.points.append(point_a)
            merger.points.append(point_b)

        # Merger starts dirty
        merger._dirty = True
        merger._render_cache = {}

        return merger

    def image_diff(self, target: Image) -> float:
        """Compute Sum of Absolute Differences (SAD) between rendered painting and target.
        Lower values indicate better fitness (fewer differences).
        """
        source = self.draw().convert("RGB")
        target_rgb = target.convert("RGB")

        # Ensure same dimensions; resize target if needed
        if source.size != target_rgb.size:
            target_rgb = target_rgb.resize(source.size, Image.BILINEAR)

        # Convert to NumPy arrays and compute SAD
        src_np = np.array(source, dtype=np.float32)
        tgt_np = np.array(target_rgb, dtype=np.float32)

        # Sum of absolute differences across all channels and pixels
        sad = float(np.sum(np.abs(src_np - tgt_np)))
        return sad
