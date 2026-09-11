# Aerial Object Detection node using YOLO26n
# Subscribes directly to Gazebo Harmonic IMX214 camera feed (with ROS 2 fallback)
# and displays real-time detections on the 'Detected Frame' window.

import os
import sys
import threading
import time
import cv2
import numpy as np

# Ensure Gazebo Python bindings and compatibility
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from ultralytics import YOLO

# Try loading Gazebo transport
HAVE_GZ = False
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
    HAVE_GZ = True
except Exception as e:
    pass

# Try loading ROS 2
HAVE_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node as RosNode
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    HAVE_ROS2 = True
except Exception as e:
    pass

DIR = os.path.dirname(os.path.abspath(__file__))
TS_PATH = os.path.join(DIR, "yolo26n.torchscript")
PT_PATH = os.path.join(DIR, "yolo26n.pt")

if os.path.exists(TS_PATH):
    MODEL_PATH = TS_PATH
    MODEL_LABEL = "YOLO26n (TorchScript)"
elif os.path.exists(PT_PATH):
    MODEL_PATH = PT_PATH
    MODEL_LABEL = "YOLO26n (PyTorch)"
else:
    MODEL_PATH = "yolo26n.pt"
    MODEL_LABEL = "YOLO26n"

print(f"[INFO] Loading model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"[INFO] {MODEL_LABEL} loaded successfully.")

_lock = threading.Lock()
_latest_frame = None
_frame_count = 0
_running = True


def on_gz_image(msg):
    global _latest_frame, _frame_count
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        with _lock:
            _latest_frame = bgr
            _frame_count += 1
    except Exception:
        pass


if HAVE_ROS2:
    class RosImageSubscriber(RosNode):
        def __init__(self):
            super().__init__("uav_camera_det")
            self.br = CvBridge()
            self.sub = self.create_subscription(
                RosImage, "camera", self.callback, 10
            )

        def callback(self, data):
            global _latest_frame, _frame_count
            try:
                bgr = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
                with _lock:
                    _latest_frame = bgr
                    _frame_count += 1
            except Exception:
                pass


def main():
    global _running

    gz_node = None
    gz_topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"

    if HAVE_GZ:
        print(f"[INFO] Subscribing to Gazebo camera: {gz_topic}")
        gz_node = GzNode()
        gz_node.subscribe(GzImage, gz_topic, on_gz_image)

    ros_executor = None
    if HAVE_ROS2:
        try:
            rclpy.init(args=None)
            ros_node = RosImageSubscriber()
            def _spin():
                try:
                    rclpy.spin(ros_node)
                except Exception:
                    pass

            ros_executor = threading.Thread(
                target=_spin, daemon=True
            )
            ros_executor.start()
            print("[INFO] Subscribed to ROS 2 topic: /camera")
        except Exception as e:
            pass

    window_name = "Detected Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    fps = 0.0
    fps_time = time.time()
    frames_processed = 0

    print("[INFO] Starting real-time detection window...")
    try:
        while _running:
            frame = None
            with _lock:
                if _latest_frame is not None:
                    frame = _latest_frame.copy()

            if frame is not None:
                # Run YOLO26n inference (class 0: person, class 2: car)
                results = model.predict(frame, classes=[0, 2], verbose=False)
                annotated = results[0].plot()

                # Calculate FPS
                frames_processed += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    fps = frames_processed / (now - fps_time)
                    frames_processed = 0
                    fps_time = now

                # Overlay status banner
                detections = len(results[0].boxes)
                status_text = f"{MODEL_LABEL} | FPS: {fps:.1f} | Detections: {detections}"
                cv2.rectangle(annotated, (10, 10), (520, 48), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    status_text,
                    (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 128),
                    2,
                )

                cv2.imshow(window_name, annotated)
            else:
                # Display waiting placeholder
                placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
                msg = "Waiting for Drone Camera Stream..."
                cv2.putText(
                    placeholder,
                    msg,
                    (200, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (180, 180, 180),
                    2,
                )
                cv2.imshow(window_name, placeholder)

            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        if HAVE_GZ and gz_node is not None:
            try:
                gz_node.unsubscribe(gz_topic)
            except Exception:
                pass
        cv2.destroyAllWindows()
        if HAVE_ROS2:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        print("[INFO] Camera detection stopped.")


if __name__ == "__main__":
    main()
