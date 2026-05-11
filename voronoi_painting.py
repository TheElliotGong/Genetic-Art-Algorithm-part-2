from random import shuffle, randint, choices, choice
from PIL import Image
import numpy as np
import cv2


class ColoredPoint:
    def __init__(self, img_width, img_height):
        self.coordinates = (randint(0, int(img_width)), randint(0, int(img_height)))
        self.color = (randint(0, 256),  # Random value for the Red channel
                      randint(0, 256),  # Random value for the Green channel
                      randint(0, 256),  # Random value for the Blue channel
                      255)              # The Alpha channel is fixed

    def __str__(self):
        return f"ColoredPoint at ({self.coordinates[0]}, {self.coordinates[1]}) of color ({self.color[0]}, {self.color[1]}, {self.color[2]})"

    def mutate(self, sigma=1.0):
        mutations = ['shift', 'color']
        weights = [50, 50]

        mutation_type = choices(mutations, weights=weights, k=1)[0]

        if mutation_type == 'shift':
            self.coordinates = (self.coordinates[0] + int(randint(-10, 10)*sigma), self.coordinates[1] + int(randint(-10, 10)*sigma))
        elif mutation_type == 'color':
            red = self.color[0] + int(randint(-25, 25)*sigma)
            green = self.color[1] + int(randint(-25, 25)*sigma)
            blue = self.color[2] + int(randint(-25, 25)*sigma)

            self.color = (red, green, blue, 255)

            # Ensure color is within correct range
            self.color = tuple(
                min(max(c, 0), 255) for c in self.color
            )


class VoronoiPainting:
    def __init__(self, num_points, target_image, background_color=(0, 0, 0)):
        self._img_width, self._img_height = target_image.size
        self.points = [ColoredPoint(self._img_width, self._img_height) for _ in range(num_points)]
        self._background_color = (*background_color, 255)
        self.target_image = target_image

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

    def mutate_points(self, rate=0.04, sigma=1.0):
        total_mutations = int(rate*self.num_points)
        random_indices = list(range(self.num_points))
        shuffle(random_indices)

        # mutate random triangles
        for i in range(total_mutations):
            index = random_indices[i]
            self.points[index].mutate(sigma=sigma)

    def shrink_points(self):
        """

        """
        self.points.pop(randint(0, self.num_points-1))

    def _render_array(self, scale=1) -> np.ndarray:
        w = self._img_width * scale
        h = self._img_height * scale

        coords = np.array([p.coordinates for p in self.points], dtype=np.int32) * scale
        xs = np.clip(coords[:, 0], 0, w - 1)
        ys = np.clip(coords[:, 1], 0, h - 1)

        binary = np.full((h, w), 255, dtype=np.uint8)
        binary[ys, xs] = 0

        _, labels = cv2.distanceTransformWithLabels(
            binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )

        colors = np.array([p.color[:3] for p in self.points], dtype=np.uint8)
        seed_labels = labels[ys, xs]
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        lut[seed_labels] = colors

        return lut[labels]

    def draw(self, scale=1) -> Image:
        return Image.fromarray(self._render_array(scale=scale), mode="RGB")
    
    def _render_array_with_lines(self, scale=1, line_width=2) -> np.ndarray:
        w = self._img_width * scale
        h = self._img_height * scale

        coords = np.array([p.coordinates for p in self.points], dtype=np.int32) * scale
        xs = np.clip(coords[:, 0], 0, w - 1)
        ys = np.clip(coords[:, 1], 0, h - 1)

        binary = np.full((h, w), 255, dtype=np.uint8)
        binary[ys, xs] = 0

        _, labels = cv2.distanceTransformWithLabels(
            binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )

        colors = np.array([p.color[:3] for p in self.points], dtype=np.uint8)
        seed_labels = labels[ys, xs]
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        lut[seed_labels] = colors

        result = lut[labels]

        label_u = np.roll(labels, -1, axis=0)
        label_d = np.roll(labels,  1, axis=0)
        label_l = np.roll(labels, -1, axis=1)
        label_r = np.roll(labels,  1, axis=1)

        is_edge = ((labels != label_u) | (labels != label_d) | (labels != label_l) | (labels != label_r))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width * 2 - 1, line_width * 2 - 1))
        is_edge = cv2.dilate(is_edge.astype(np.uint8), kernel).astype(bool)

        result[is_edge] = 0

        return result
    
    def draw_lines(self, scale=1, line_width=2) -> Image:
        return Image.fromarray(self._render_array_with_lines(scale=scale, line_width=line_width), mode="RGB")


    @staticmethod
    def _mate_possible(a, b) -> bool:
        return all([a.num_points == b.num_points,
                   a.get_img_width == b.get_img_width,
                   a.get_img_height == b.get_img_height])

    @staticmethod
    def mate(a, b):
        if not VoronoiPainting._mate_possible(a, b):
            if a.num_points > b.num_points:
                return a, a
            else:
                return b, b

        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i])/2) for i in range(3))

        child_a = VoronoiPainting(0, a.target_image, background_color=new_background)
        child_b = VoronoiPainting(0, a.target_image, background_color=new_background)

        for point_a, point_b in zip(a.points, b.points):
            if randint(0, 1) == 0:
                child_a.points.append(point_a)
                child_b.points.append(point_b)
            else:
                child_a.points.append(point_b)
                child_b.points.append(point_a)

        return child_a, child_b

    @staticmethod
    def merge(a, b):
        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i])/2) for i in range(3))

        merger = VoronoiPainting(0, a.target_image, background_color=new_background)

        for point_a, point_b in zip(a.points, b.points):
            merger.points.append(point_a)
            merger.points.append(point_b)

        return merger

    def image_diff(self, target: Image) -> float:
        if not hasattr(self, "_target_np") or self._target_np is None:
            self._target_np = np.array(target.convert("RGB"), dtype=np.uint8)

        source = self._render_array()
        return cv2.norm(source, self._target_np, cv2.NORM_L1)

