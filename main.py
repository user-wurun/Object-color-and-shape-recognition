import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from color_analysis import estimate_object_color, gray_world_white_balance, print_lab_value
from color_value import REF_LAB
from shape_cnn import load_model, predict_shape

WINDOW_NAME = "Camera - CNN Shape Recognition"
COLOR_NAMES = {
	"red": "红色",
	"orange": "橙色",
	"yellow": "黄色",
	"green": "绿色",
	"cyan": "青色",
	"blue": "蓝色",
	"purple": "紫色",
	"black": "黑色",
	"white": "白色",
	"gray": "灰色",
}


def draw_chinese_text(image, text, position, color=(0, 255, 0), font_size=28):
	"""Draw Chinese text with a Windows font; fall back to a default font if unavailable."""
	font_paths = (
		"C:/Windows/Fonts/msyh.ttc",
		"C:/Windows/Fonts/simhei.ttf",
		"C:/Windows/Fonts/simsun.ttc",
	)
	font = None
	for font_path in font_paths:
		try:
			font = ImageFont.truetype(font_path, font_size)
			break
		except OSError:
			continue

	if font is None:
		cv2.putText(image, text.encode("ascii", "replace").decode(), position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		return image

	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pil_image = Image.fromarray(rgb_image)
	draw = ImageDraw.Draw(pil_image)
	draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
	return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def find_object_contours(frame):
	"""Find all saturated foreground objects while rejecting wood background."""
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# Keep saturated yellow separately; its hue overlaps the brown wood background.
	saturated = hsv[:, :, 1] >= 45
	bright = hsv[:, :, 2] >= 65
	non_wood_hue = (hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 27)
	yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55)
	mask = np.where(saturated & bright & (non_wood_hue | yellow), 255, 0).astype(np.uint8)
	kernel = np.ones((7, 7), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	frame_height, frame_width = mask.shape
	frame_area = frame_height * frame_width
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	candidates = []
	for contour in contours:
		area = cv2.contourArea(contour)
		x, y, width, height = cv2.boundingRect(contour)
		if area < frame_area * 0.01 or area > frame_area * 0.85:
			continue
		if x <= 1 or y <= 1 or x + width >= frame_width - 1 or y + height >= frame_height - 1:
			continue
		candidates.append(contour)
	return sorted(candidates, key=cv2.contourArea, reverse=True)


def find_object_contour(frame):
	"""Return the largest object for backwards-compatible tests."""
	contours = find_object_contours(frame)
	return contours[0] if contours else None

def main():
	model = load_model()
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		raise RuntimeError("无法打开摄像头，请检查摄像头连接或权限。")

	data = {
		"lab_image": None,
		"point": None,
		"reference_colors": REF_LAB,
	}
	cv2.namedWindow(WINDOW_NAME)
	cv2.setMouseCallback(WINDOW_NAME, print_lab_value, data)

	try:
		while True:
			success, frame = camera.read()
			if not success:
				print("读取摄像头画面失败。")
				break

			color_frame = gray_world_white_balance(frame)
			# VideoCapture returns BGR. Convert explicitly through RGB before Lab.
			rgb_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
			data["lab_image"] = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
			color_hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

			display = frame.copy()
			for contour in find_object_contours(color_frame):
				shape_name, confidence = predict_shape(model, contour, frame)
				color_name, color_probabilities, sample_points = estimate_object_color(
					data["lab_image"],
					contour,
					data["reference_colors"],
					seed=hash(cv2.boundingRect(contour)),
					hsv_image=color_hsv,
				)
				x, y, width, height = cv2.boundingRect(contour)
				cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
				if color_name is not None:
					color_text = COLOR_NAMES.get(color_name, color_name)
					color_probability = color_probabilities[color_name]
					label = f"形状: {shape_name} ({confidence:.0%}) 颜色: {color_text} ({color_probability:.0%})"
					display = draw_chinese_text(display, label, (x, max(5, y - 38)))
					for point_x, point_y in sample_points:
						cv2.circle(display, (point_x, point_y), 2, (255, 255, 255), -1)
			if data["point"] is not None:
				point_x, point_y = data["point"]
				cv2.drawMarker(
					display,
					(point_x, point_y),
					(0, 0, 255),
					cv2.MARKER_CROSS,
					20,
					2,
				)

			cv2.imshow(WINDOW_NAME, display)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
	finally:
		camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
