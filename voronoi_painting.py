from random import shuffle, randint, choices, choice
from PIL import Image
import numpy as np
import cv2


# This class represents a single point in the Voronoi painting, with both coordinates and color information.
# It includes methods for mutating the point's position or color, which are used during the evolution process to create variations in the paintings.
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
        """
        Mutate the point's coordinates or color with a given mutation rate and sigma for the mutation strength.
        :param sigma: The standard deviation for the mutation strength. Higher values will result in more significant mutations.
        """
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


# This class represents a Voronoi painting, which consists of a collection of colored points that define the regions of the painting.
class VoronoiPainting:
    def __init__(self, num_points, target_image, background_color=(0, 0, 0)):
        self._img_width, self._img_height = target_image.size
        self.points = [
            ColoredPoint(self._img_width, self._img_height) for _ in range(num_points)
        ]
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
        """Mutate a percentage of the points in the painting based on the given mutation rate and sigma for mutation strength.
        :param rate: The percentage of points to mutate (between 0 and 1).
        :param sigma: The standard deviation for the mutation strength. Higher values will result in more
        """
        total_mutations = int(rate * self.num_points)
        random_indices = list(range(self.num_points))
        shuffle(random_indices)

        # mutate random triangles
        for i in range(total_mutations):
            index = random_indices[i]
            self.points[index].mutate(sigma=sigma)

    def shrink_points(self):
        """This function removes a random point from the painting, effectively shrinking the painting by one point. It modifies the painting in place."""
        self.points.pop(randint(0, self.num_points - 1))

    def _render_array(self, scale=1) -> np.ndarray:
        """This function renders the Voronoi painting as a NumPy array.
        It creates a binary image where the points are marked, then uses the distance transform to assign each pixel to the nearest point,
        and finally creates a color lookup table to generate the final image based on the colors of the points.
        """
        # Find the dimensions of the output image based on the original image dimensions and the scale factor.
        w = self._img_width * scale
        h = self._img_height * scale
        # Create a binary image where the coordinates of the points are marked as 0 (black) and the rest of the pixels are 255 (white).
        coords = np.array([p.coordinates for p in self.points], dtype=np.int32) * scale
        xs = np.clip(coords[:, 0], 0, w - 1)
        ys = np.clip(coords[:, 1], 0, h - 1)
        # Use the distance transform to assign each pixel to the nearest point, which creates a Voronoi diagram. The labels from the distance transform indicate which point is closest to each pixel.
        binary = np.full((h, w), 255, dtype=np.uint8)
        binary[ys, xs] = 0
        # Create a color lookup table (LUT) where the index corresponds to the label from the distance transform, and the value is the color of the corresponding point.
        # The final image is generated by mapping each pixel's label to its corresponding color in the LUT.
        _, labels = cv2.distanceTransformWithLabels(
            binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Extract the RGB colors from the points and create a LUT based on the labels assigned to the seed points. The LUT is then used to generate the final image array.
        colors = np.array([p.color[:3] for p in self.points], dtype=np.uint8)
        seed_labels = labels[ys, xs]
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        lut[seed_labels] = colors

        return lut[labels]

    def draw(self, scale=1) -> Image:
        return Image.fromarray(self._render_array(scale=scale), mode="RGB")

    def _render_array_with_lines(
        self, scale=1, line_width=2, line_color=(0, 0, 0)
    ) -> np.ndarray:
        """This function renders the Voronoi painting as a NumPy array with lines between the regions.
        It creates a binary image where the points are marked, then uses the distance transform to assign each pixel to the nearest point,
        and finally creates a color lookup table to generate the final image based on the colors of the points.
        It also identifies the edges between regions and draws lines of specified width and color along those edges.
        param scale: The scale factor for the output image.
        param line_width: The width of the lines between the regions.
        param line_color: The color of the lines between the regions.
        """
        # Find the dimensions of the output image based on the original image dimensions and the scale factor.
        w = self._img_width * scale
        h = self._img_height * scale
        # Create a binary image where the coordinates of the points are marked as 0 (black) and the rest of the pixels are 255 (white).
        coords = np.array([p.coordinates for p in self.points], dtype=np.int32) * scale
        xs = np.clip(coords[:, 0], 0, w - 1)
        ys = np.clip(coords[:, 1], 0, h - 1)
        # Use the distance transform to assign each pixel to the nearest point, which creates a Voronoi diagram. The labels from the distance transform indicate which point is closest to each pixel.
        binary = np.full((h, w), 255, dtype=np.uint8)
        binary[ys, xs] = 0
        # Create a color lookup table (LUT) where the index corresponds to the label from the distance transform, and the value is the color of the corresponding point.
        _, labels = cv2.distanceTransformWithLabels(
            binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        # Extract the RGB colors from the points and create a LUT based on the labels assigned to the seed points. The LUT is then used to generate the final image array.
        colors = np.array([p.color[:3] for p in self.points], dtype=np.uint8)
        seed_labels = labels[ys, xs]
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        lut[seed_labels] = colors

        result = lut[labels]
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
    def mate(a, b):
        """Mate two paintings by combining their points. If the paintings have different numbers of points or different image dimensions, the one with more points will be duplicated to create a child painting.
        Otherwise, a new child painting is created by randomly selecting points from both parents.
        """
        # Check if mating is possible based on the number of points and image dimensions of the two paintings.
        # If they are not compatible for mating, the painting with more points is duplicated to create a child painting, while the other painting is returned unchanged.
        if not VoronoiPainting._mate_possible(a, b):
            if a.num_points > b.num_points:
                return a, a
            else:
                return b, b
        # If mating is possible, a new child painting is created by randomly selecting points from both parents.
        # The background color of the child painting is set to the average of the background colors of the two parent paintings.
        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i]) / 2) for i in range(3))
        # Create two child paintings with the same target image and background color. The points of the child paintings are then filled by randomly selecting points
        #  from either parent for each corresponding point index.
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
        """Merge two paintings by combining all their points. The background color of the merged painting is the average of the background colors of the two input paintings.
        The resulting painting will have twice the number of points as the input paintings, as it combines all points from both paintings.
        """
        # Calculate the average background color by taking the RGB values of both paintings and averaging them. This will be used as the background color for the merged painting.
        ab = a.get_background_color
        bb = b.get_background_color
        new_background = (int((ab[i] + bb[i]) / 2) for i in range(3))
        # Create a new VoronoiPainting with the same target image and the calculated average background color. The points of the merged painting are filled by combining all points from both input paintings.
        merger = VoronoiPainting(0, a.target_image, background_color=new_background)

        for point_a, point_b in zip(a.points, b.points):
            merger.points.append(point_a)
            merger.points.append(point_b)

        return merger

    def image_diff(self, target: Image) -> float:
        """Check the difference between the rendered painting and the target image using L1 norm.
        The target image is converted to a NumPy array if it hasn't been already, and the rendered painting is also converted to a NumPy array.
        The L1 norm is then calculated between the two arrays to measure the difference.
        """
        if not hasattr(self, "_target_np") or self._target_np is None:
            self._target_np = np.array(target.convert("RGB"), dtype=np.uint8)

        source = self._render_array()
        return cv2.norm(source, self._target_np, cv2.NORM_L1)
