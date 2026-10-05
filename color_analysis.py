import cv2
import numpy as np
import random


def gray_world_white_balance(image):
	"""Reduce camera color cast using the frame's average channel intensity."""
	image_float = image.astype(np.float32)
	channel_means = image_float.reshape(-1, 3).mean(axis=0)
	target = float(channel_means.mean())
	gains = np.clip(target / np.maximum(channel_means, 1.0), 0.65, 1.5)
	return np.clip(image_float * gains, 0, 255).astype(np.uint8)


def _hsv_color_name(hue, saturation, value):
	if value < 45:
		return "black"
	if saturation < 32:
		return "white" if value >= 180 else "gray"
	if hue <= 8 or hue >= 170:
		return "red"
	if hue <= 18:
		return "orange"
	if hue <= 42:
		return "yellow"
	if hue <= 85:
		return "green"
	if hue <= 103:
		return "cyan"
	if hue <= 135:
		return "blue"
	return "purple"


def find_closest_color(lab_value, reference_colors):
	"""Return the reference color with the smallest CIE76 Delta E."""
	closest_name = None
	closest_delta_e = float("inf")

	for name, reference_lab in reference_colors.items():
		delta_e = float(np.linalg.norm(lab_value - reference_lab))
		if delta_e < closest_delta_e:
			closest_name = name
			closest_delta_e = delta_e

	return closest_name, closest_delta_e


def estimate_object_color(lab_image, contour, reference_colors, sample_count=40, seed=None, hsv_image=None):
	"""Estimate an object's color by voting over random points inside its contour."""
	mask = np.zeros(lab_image.shape[:2], dtype=np.uint8)
	cv2.drawContours(mask, [contour], -1, 255, -1)
	mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
	y_coordinates, x_coordinates = np.where(mask > 0)
	if len(x_coordinates) == 0:
		return None, {}, []

	random_generator = random.Random(seed)
	point_count = min(sample_count, len(x_coordinates))
	selected_indices = random_generator.sample(range(len(x_coordinates)), point_count)
	votes = {name: 0 for name in reference_colors}
	points = []
	for index in selected_indices:
		x = int(x_coordinates[index])
		y = int(y_coordinates[index])
		if hsv_image is not None:
			hue, saturation, value = hsv_image[y, x]
			color_name = _hsv_color_name(int(hue), int(saturation), int(value))
		else:
			lab_l, lab_a, lab_b = lab_image[y, x]
			lab_value = np.array([
				float(lab_l) * 100.0 / 255.0,
				float(lab_a) - 128.0,
				float(lab_b) - 128.0,
			])
			color_name, _ = find_closest_color(lab_value, reference_colors)
		votes[color_name] += 1
		points.append((x, y))

	total_votes = sum(votes.values())
	probabilities = {
		name: count / total_votes for name, count in votes.items() if count > 0
	}
	color_name = max(votes, key=votes.get)
	return color_name, probabilities, points

def print_lab_value(event, x, y, _flags, data):
	"""Print the Lab value at the clicked pixel."""
	if event != cv2.EVENT_LBUTTONDOWN:
		return

	lab_image = data["lab_image"]
	if lab_image is None:
		return

	height, width = lab_image.shape[:2]
	if not (0 <= x < width and 0 <= y < height):
		return

	lab_l, lab_a, lab_b = lab_image[y, x]

	# OpenCV's 8-bit Lab stores L in 0-255 and offsets a/b by 128.
	lightness = float(lab_l) * 100.0 / 255.0
	a_value = float(lab_a) - 128.0
	b_value = float(lab_b) - 128.0
	lab_value = np.array([lightness, a_value, b_value])
	color_name, delta_e = find_closest_color(lab_value, data["reference_colors"])

	data["point"] = (x, y)
	print(
		f"点击坐标 ({x}, {y}) -> "
		f"Lab=({lightness:.2f}, {a_value:.2f}, {b_value:.2f}), "
		f"目标颜色={color_name}, Delta E={delta_e:.2f}"
	)
