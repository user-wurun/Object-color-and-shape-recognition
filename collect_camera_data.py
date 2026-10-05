from datetime import datetime
from pathlib import Path

import cv2

from main import draw_chinese_text, find_object_contour
from shape_cnn import CLASS_NAMES, DATASET_PATH


WINDOW_NAME = "采集电脑摄像头训练数据"
KEY_TO_CLASS = {
    ord("1"): (0, "球"),
    ord("2"): (1, "三棱柱"),
    ord("3"): (2, "圆柱"),
    ord("4"): (3, "圆锥"),
    ord("5"): (4, "长方体"),
    ord("6"): (5, "正方体"),
}


def save_crop(frame, rectangle, class_name, counter):
    x, y, width, height = rectangle
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], x + width)
    y1 = min(frame.shape[0], y + height)
    crop = frame[y0:y1, x0:x1]

    output_dir = DATASET_PATH / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"camera_{timestamp}_{counter:04d}.jpg"
    success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("训练图片编码失败。")
    encoded.tofile(output_path)
    return output_path


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("无法打开电脑摄像头，请检查摄像头是否被其他程序占用。")

    counts = {class_name: 0 for class_name in CLASS_NAMES}
    counter = 0
    manual_roi = None
    manual_frame = None
    cv2.namedWindow(WINDOW_NAME)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("读取摄像头画面失败。")
                break

            display = frame.copy()
            contour = find_object_contour(frame)
            if manual_roi is not None:
                x, y, width, height = manual_roi
                cv2.rectangle(display, (x, y), (x + width, y + height), (255, 0, 0), 2)
                display = draw_chinese_text(display, "手动框选模式", (x, max(5, y - 38)), (255, 0, 0))
            elif contour is None:
                display = draw_chinese_text(display, "未检测到物块，请将物块放在画面中央", (20, 40), (0, 0, 255))
            else:
                x, y, width, height = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
                display = draw_chinese_text(
                    display,
                    f"物块大小: {width} x {height} px, 面积: {area:.0f}",
                    (x, max(5, y - 38)),
                )

            instruction = "R手动框选 C清除框  1球 2三棱柱 3圆柱 4圆锥 5长方体 6正方体 Q退出"
            display = draw_chinese_text(display, instruction, (20, frame.shape[0] - 45), (255, 255, 0), 24)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                manual_frame = frame.copy()
                selected_roi = cv2.selectROI(
                    WINDOW_NAME,
                    manual_frame,
                    fromCenter=False,
                    showCrosshair=True,
                )
                if selected_roi[2] > 0 and selected_roi[3] > 0:
                    manual_roi = tuple(int(value) for value in selected_roi)
                    print(
                        f"已设置手动框: x={manual_roi[0]}, y={manual_roi[1]}, "
                        f"width={manual_roi[2]}, height={manual_roi[3]}"
                    )
                else:
                    manual_frame = None
                continue
            if key == ord("c"):
                manual_roi = None
                manual_frame = None
                print("已清除手动框，恢复自动检测。")
                continue
            if key not in KEY_TO_CLASS or (manual_roi is None and contour is None):
                continue

            _, class_name = KEY_TO_CLASS[key]
            if manual_roi is not None:
                rectangle = manual_roi
                source_frame = manual_frame
            else:
                if contour is None:
                    continue
                rectangle = cv2.boundingRect(contour)
                source_frame = frame
            if source_frame is None:
                continue
            output_path = save_crop(source_frame, rectangle, class_name, counter)
            if manual_roi is not None:
                manual_roi = None
                manual_frame = None
                print("本次手动框已保存，下一帧恢复自动检测。")
            counts[class_name] += 1
            counter += 1
            print(f"已保存: {output_path}，标签={class_name}，当前类别数量={counts[class_name]}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("本次采集统计:")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} 张")


if __name__ == "__main__":
    main()
