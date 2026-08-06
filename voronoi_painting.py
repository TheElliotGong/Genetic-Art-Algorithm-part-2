from copy import deepcopy
from random import random, randint, sample
from PIL import Image
import numpy as np
import cv2

import target_cache


# Scratch buffers reused across renders, keyed by output shape. Rendering happens
# once per individual per generation, so allocating four full-size arrays every
# call is pure overhead. The buffers are module-level (one set per process) and
# are only ever touched inside a single render call, which is safe because
# ``evol`` parallelizes with processes rather than threads.
_RENDER_BUFFERS = {}

# Tiled runs use a handful of distinct tile sizes; the cap keeps the buffer
# cache from growing without bound if many sizes are rendered.
_MAX_BUFFER_SHAPES = 8


def _render_buffers(height, width):
    """Return reusable scratch arrays for rendering at the given size.

    :param height: Output height in pixels.
    :param width: Output width in pixels.
    """
    key = (height, width)
    buffers = _RENDER_BUFFERS.get(key)
    if buffers is None:
        if len(_RENDER_BUFFERS) >= _MAX_BUFFER_SHAPES:
            _RENDER_BUFFERS.clear()
        buffers = (
            np.empty((height, width), dtype=np.uint8),  # binary seed image
            np.empty((height, width), dtype=np.float32),  # distance transform
            np.empty((height, width), dtype=np.int32),  # nearest-seed labels
            np.empty((height, width, 3), dtype=np.uint8),  # rendered pixels
        )
        _RENDER_BUFFERS[key] = buffers
    return buffers


# This class represents a single point in the Voronoi painting, with both coordinates and color information.
# It includes methods for mutating the point's position or color, which are used during the evolution process to create variations in the paintings.
class ColoredPoint:
    # __slots__ keeps points small and makes attribute access, copying and
    # pickling faster. Populations hold hundreds of thousands of these.
    __slots__ = ("coordinates", "color")

    def __init__(self, img_width, img_height):
        self.coordinates = (randint(0, int(img_width)), randint(0, int(img_height)))
        self.color = (
            randint(0, 255),  # Random value for the Red channel
            randint(0, 255),  # Random value for the Green channel
            randint(0, 255),  # Random value for the Blue channel
            255,
        )  # The Alpha channel is fixed

    def __str__(self):
        return f"ColoredPoint at ({self.coordinates[0]}, {self.coordinates[1]}) of color ({self.color[0]}, {self.color[1]}, {self.color[2]})"

    def copy(self):
        """Return an independent copy of this point.

        Coordinates and colors are immutable tuples that mutation replaces
        rather than edits in place, so copying the two references is enough to
        fully detach the copy from the original. This is the cheap replacement
        for ``deepcopy`` in the evolution operators.
        """
        clone = ColoredPoint.__new__(ColoredPoint)
        clone.coordinates = self.coordinates
        clone.color = self.color
        return clone

    def __getstate__(self):
        return (self.coordinates, self.color)

    def __setstate__(self, state):
        self.coordinates, self.color = state

    def __deepcopy__(self, memo):
        clone = self.copy()
        memo[id(self)] = clone
        return clone

    def mutate(self, sigma=1.0):
        """
        Mutate the point's coordinates or color with a given mutation rate and sigma for the mutation strength.
        :param sigma: The standard deviation for the mutation strength. Higher values will result in more significant mutations.
        """
        # An even coin flip between the two mutation types. random() is used
        # directly because random.choices rebuilds its cumulative weights on
        # every call, which is a measurable cost at this call frequency.
        if random() < 0.5:
            self.coordinates = (
                self.coordinates[0] + int(randint(-10, 10) * sigma),
                self.coordinates[1] + int(randint(-10, 10) * sigma),
            )
        else:
            red = self.color[0] + int(randint(-25, 25) * sigma)
            green = self.color[1] + int(randint(-25, 25) * sigma)
            blue = self.color[2] + int(randint(-25, 25) * sigma)

            self.color = (red, green, blue, 255)

            # Ensure color is within correct range
            self.color = tuple(min(max(c, 0), 255) for c in self.color)


# This class represents a Voronoi painting, which consists of a collection of colored points that define the regions of the painting.
class VoronoiPainting:
    def __init__(self, num_points, target_image, background_color=(0, 0, 0)):
        self._img_width, self._img_height = target_image.size
        self.points = [
            ColoredPoint(self._img_width, self._img_height) for _ in range(num_points)
        ]
        self._background_color = (*background_color, 255)
        # The target pixels live in a per-process registry rather than on the
        # painting, so copying or serializing a painting never copies the image.
        self._target_key = target_cache.register(target_image)

    @property
    def target_image(self):
        """The target image this painting is evolving towards."""
        return target_cache.image_for(self._target_key)

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
    def num_points(self):
        return len(self.points)

    def __repr__(self):
        return "VoronoiPainting with %d triangles" % self.num_points

    def __getstate__(self):
        """Serialize without the target image, and with the genome packed flat.

        Only the genome travels between processes; the target pixels are
        resolved from the registry on the other side. When the registry cannot
        hand the target to another process (no spool file available), the raw
        pixels are embedded as a fallback so serialization is never lossy.

        The points are packed into two arrays rather than sent as a list of
        objects. ``evol`` serializes with dill, which walks and dispatches on
        every object in the graph; handing it two arrays instead of hundreds of
        points is an order of magnitude less work per individual per generation.
        """
        coordinates = np.array(
            [point.coordinates for point in self.points], dtype=np.int32
        ).reshape(-1, 2)
        colors = np.array(
            [point.color for point in self.points], dtype=np.uint8
        ).reshape(-1, 4)

        return {
            "width": self._img_width,
            "height": self._img_height,
            "background": self._background_color,
            "coordinates": coordinates,
            "colors": colors,
            "target_key": self._target_key,
            "target_payload": (
                None
                if target_cache.is_portable(self._target_key)
                else target_cache.payload_for(self._target_key)
            ),
        }

    def __setstate__(self, state):
        self._img_width = state["width"]
        self._img_height = state["height"]
        self._background_color = state["background"]
        self._target_key = state["target_key"]

        # tolist() converts to Python scalars in one pass, which is much faster
        # than indexing the arrays element by element while rebuilding points.
        points = []
        for coordinates, color in zip(
            state["coordinates"].tolist(), state["colors"].tolist()
        ):
            point = ColoredPoint.__new__(ColoredPoint)
            point.coordinates = (coordinates[0], coordinates[1])
            point.color = (color[0], color[1], color[2], color[3])
            points.append(point)
        self.points = points

        # Resolve now rather than lazily so that a missing target fails loudly
        # at deserialization time instead of deep inside a worker's scoring call.
        target_cache.resolve(self._target_key, payload=state.get("target_payload"))

    def __deepcopy__(self, memo):
        """Copy the genome only; the target image is shared, never duplicated."""
        clone = VoronoiPainting.__new__(VoronoiPainting)
        clone._img_width = self._img_width
        clone._img_height = self._img_height
        clone._background_color = self._background_color
        clone._target_key = self._target_key
        clone.points = [point.copy() for point in self.points]
        memo[id(self)] = clone
        return clone

    def mutate_points(self, rate=0.04, sigma=1.0):
        """Mutate a percentage of the points in the painting based on the given mutation rate and sigma for mutation strength.
        :param rate: The percentage of points to mutate (between 0 and 1).
        :param sigma: The standard deviation for the mutation strength. Higher values will result in more
        """
        total_mutations = int(rate * self.num_points)
        if total_mutations <= 0:
            return

        # Sampling the indices to mutate is O(k); shuffling every index to then
        # use the first few is O(n) and dominates at low mutation rates.
        for index in sample(range(self.num_points), total_mutations):
            self.points[index].mutate(sigma=sigma)

    def shrink_points(self):
        """This function removes a random point from the painting, effectively shrinking the painting by one point. It modifies the painting in place."""
        self.points.pop(randint(0, self.num_points - 1))

    def _render_into_buffer(self, scale=1):
        """Render the painting into the shared scratch buffer for its size.

        Returns ``(pixels, labels)`` where ``pixels`` is a view on a reused
        buffer that stays valid only until the next render in this process.
        Callers that keep the result must copy it; the scoring path does not,
        which is what makes this the cheap path.

        :param scale: The scale factor for the output image.
        """
        # Find the dimensions of the output image based on the original image dimensions and the scale factor.
        w = self._img_width * scale
        h = self._img_height * scale
        binary, distances, labels, output = _render_buffers(h, w)
        # Create a binary image where the coordinates of the points are marked as 0 (black) and the rest of the pixels are 255 (white).
        coords = np.array([p.coordinates for p in self.points], dtype=np.int32) * scale
        xs = np.clip(coords[:, 0], 0, w - 1)
        ys = np.clip(coords[:, 1], 0, h - 1)
        # Refilling the reused buffer is a plain memset, which is cheaper than
        # allocating a fresh array on every render.
        binary.fill(255)
        binary[ys, xs] = 0
        # Use the distance transform to assign each pixel to the nearest point, which creates a Voronoi diagram. The labels from the distance transform indicate which point is closest to each pixel.
        cv2.distanceTransformWithLabels(
            binary,
            cv2.DIST_L2,
            3,
            dst=distances,
            labels=labels,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        # Create a color lookup table (LUT) where the index corresponds to the label from the distance transform, and the value is the color of the corresponding point.
        # The final image is generated by mapping each pixel's label to its corresponding color in the LUT.
        colors = np.array([p.color[:3] for p in self.points], dtype=np.uint8)
        seed_labels = labels[ys, xs]
        # Every label in the image also appears at a seed pixel, so the LUT can
        # be sized from the seeds instead of scanning the whole label image.
        lut = np.zeros((int(seed_labels.max()) + 1, 3), dtype=np.uint8)
        lut[seed_labels] = colors

        # np.take writing into a preallocated buffer is substantially faster
        # than fancy indexing, which has to allocate its result each time.
        np.take(lut, labels, axis=0, out=output)

        return output, labels

    def _render_array(self, scale=1) -> np.ndarray:
        """This function renders the Voronoi painting as a NumPy array.
        It creates a binary image where the points are marked, then uses the distance transform to assign each pixel to the nearest point,
        and finally creates a color lookup table to generate the final image based on the colors of the points.
        The returned array is an independent copy, safe to hold on to.
        """
        pixels, _ = self._render_into_buffer(scale=scale)
        return pixels.copy()

    def draw(self, scale=1) -> Image:
        return Image.fromarray(self._render_array(scale=scale), mode="RGB")

    def _render_array_with_lines(
        self, scale=1, line_width=2, line_color=(0, 0, 0)
    ) -> np.ndarray:
        """This function renders the Voronoi painting as a NumPy array with lines between the regions.
        It renders the painting, then identifies the edges between regions and draws lines of specified
        width and color along those edges.
        param scale: The scale factor for the output image.
        param line_width: The width of the lines between the regions.
        param line_color: The color of the lines between the regions.
        """
        pixels, labels = self._render_into_buffer(scale=scale)
        # This result is returned to the caller and outlives the shared buffer,
        # so it has to be a copy.
        result = pixels.copy()
        # Identify edges between regions by comparing the labels of neighboring pixels. If a pixel has a different label than any of its neighbors, it is considered an edge.
        # The edges are then dilated to create lines of the specified width, and the line color is applied to those edge pixels in the final image array.
        label_u = np.roll(labels, -1, axis=0)
        label_d = np.roll(labels, 1, axis=0)
        label_l = np.roll(labels, -1, axis=1)
        label_r = np.roll(labels, 1, axis=1)

        is_edge = (
            (labels != label_u)
            | (labels != label_d)
            | (labels != label_l)
            | (labels != label_r)
        )
        # Dilate the edges to create lines of the specified width. The dilation is done using a structuring element (kernel) that defines the shape and size of the dilation.
        #  The resulting dilated edges are then colored with the specified line color in the final image array.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (line_width * 2 - 1, line_width * 2 - 1)
        )
        is_edge = cv2.dilate(is_edge.astype(np.uint8), kernel).astype(bool)

        result[is_edge] = line_color

        return result

    def draw_lines(self, scale=1, line_width=2, line_color=(0, 0, 0)) -> Image:
        """
        Draw the Voronoi painting with lines between the regions.
        :param scale: The scale factor for the output image.
        :param line_width: The width of the lines between the regions.
        :param line_color: The color of the lines between the regions."""
        return Image.fromarray(
            self._render_array_with_lines(
                scale=scale, line_width=line_width, line_color=line_color
            ),
            mode="RGB",
        )

    @staticmethod
    def _mate_possible(a, b) -> bool:
        """Check if two paintings can be mated based on their number of points and image dimensions.
         Two paintings can be mated if they have the same number of points and the same image dimensions. If they differ in either of these aspects, mating is not possible.
        :param a: The first VoronoiPainting.
        :param b: The second VoronoiPainting."""
        return all(
            [
                a.num_points == b.num_points,
                a.get_img_width == b.get_img_width,
                a.get_img_height == b.get_img_height,
            ]
        )

    @staticmethod
    def _blend_background(a, b):
        """Average the background colors of two paintings."""
        ab = a.get_background_color
        bb = b.get_background_color
        return (int((ab[i] + bb[i]) / 2) for i in range(3))

    @staticmethod
    def breed(a, b):
        """Produce a single child by uniform crossover of two paintings.

        This is the operator the evolution loop actually uses. The child owns
        fresh copies of the selected points, so no further copying is needed
        before it is mutated.

        :param a: The first parent painting.
        :param b: The second parent painting.
        """
        # Check if mating is possible based on the number of points and image dimensions of the two paintings.
        # If they are not compatible, the painting with more points is copied.
        if not VoronoiPainting._mate_possible(a, b):
            return deepcopy(a if a.num_points > b.num_points else b)

        child = VoronoiPainting(
            0, a.target_image, background_color=VoronoiPainting._blend_background(a, b)
        )
        # For each point index, inherit from one parent or the other.
        for point_a, point_b in zip(a.points, b.points):
            child.points.append(point_a.copy() if randint(0, 1) == 0 else point_b.copy())

        return child

    @staticmethod
    def mate(a, b):
        """Mate two paintings by combining their points. If the paintings have different numbers of points or different image dimensions, the one with more points will be duplicated to create a child painting.
        Otherwise, two child paintings are created by randomly selecting points from both parents. Both children own copies of their points.
        """
        # Check if mating is possible based on the number of points and image dimensions of the two paintings.
        # If they are not compatible for mating, the painting with more points is duplicated to create a child painting, while the other painting is returned unchanged.
        if not VoronoiPainting._mate_possible(a, b):
            if a.num_points > b.num_points:
                return a, a
            else:
                return b, b
        # If mating is possible, two new child paintings are created by randomly selecting points from both parents.
        # The background color of the child paintings is set to the average of the background colors of the two parent paintings.
        new_background = VoronoiPainting._blend_background(a, b)
        # Create two child paintings with the same target image and background color. The points of the child paintings are then filled by randomly selecting points
        #  from either parent for each corresponding point index.
        child_a = VoronoiPainting(0, a.target_image, background_color=new_background)
        child_b = VoronoiPainting(0, a.target_image, background_color=new_background)

        for point_a, point_b in zip(a.points, b.points):
            if randint(0, 1) == 0:
                child_a.points.append(point_a.copy())
                child_b.points.append(point_b.copy())
            else:
                child_a.points.append(point_b.copy())
                child_b.points.append(point_a.copy())

        return child_a, child_b

    @staticmethod
    def merge(a, b):
        """Merge two paintings by combining all their points. The background color of the merged painting is the average of the background colors of the two input paintings.
        The resulting painting will have twice the number of points as the input paintings, as it combines all points from both paintings.
        The merged painting owns copies of the points, so it is safe to mutate.
        """
        # Create a new VoronoiPainting with the same target image and the averaged background color. The points of the merged painting are filled by combining all points from both input paintings.
        merger = VoronoiPainting(
            0, a.target_image, background_color=VoronoiPainting._blend_background(a, b)
        )

        for point_a, point_b in zip(a.points, b.points):
            merger.points.append(point_a.copy())
            merger.points.append(point_b.copy())

        return merger

    def image_diff(self, target: Image) -> float:
        """Check the difference between the rendered painting and the target image using L1 norm.
        The target's RGB pixels are cached per process by the target registry, so
        the conversion happens once per run rather than once per painting.
        """
        target_np = target_cache.rgb_of(target)
        # The scoring path consumes the rendered pixels immediately, so it can
        # read straight from the shared buffer without copying.
        source, _ = self._render_into_buffer()
        return cv2.norm(source, target_np, cv2.NORM_L1)
