from ultralytics import YOLO

import numpy as np, cv2


class TargetModel:
    def __init__(self):
        self.model = YOLO("models/target_best.pt")

    def order_points(self, pts):
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype="float32"
        )

    def predict(self, frame):
        result = self.model(frame, conf=0.8, verbose=False)[0]

        if result.masks is None:
            return None

        for mask in result.masks.xy:
            pts = np.array(mask, dtype=np.int32)

            epsilon = 0.04 * cv2.arcLength(pts, True)
            approx = cv2.approxPolyDP(pts, epsilon, True)

            corners = approx.reshape(-1, 2).tolist()

            if len(corners) == 4:
                ordered = self.order_points(corners)
                return ordered.tolist()

        return None
