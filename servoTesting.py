import os
import sys
import time
import cv2
import numpy as np

from gpiozero import AngularServo
from ultralytics import YOLO

### USER PARAMETERS ###
model_path = 'my_model_ncnn_model'
cam_source = 'usb0'

tra_min_thresh = 0.4
rec_min_thresh = 0.7

resW, resH = 640, 640

STAY_OPEN = 5.0
COOLDOWN = 2.0
STARTUP_DELAY = 2.0

### SERVO SETUP ###
trash_servo = AngularServo(
    14,
    min_angle=-90,
    max_angle=90,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.4/1000
)

recycle_servo = AngularServo(
    15,
    min_angle=-90,
    max_angle=90,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.4/1000
)

# --- SERVO HELPER (critical fix) ---
def move_servo(servo, angle):
    servo.detach()
    time.sleep(0.05)
    servo.angle = angle

# --- SAFE STARTUP ---
move_servo(trash_servo, 0)
move_servo(recycle_servo, 0)

time.sleep(1.0)

trash_servo.detach()
recycle_servo.detach()

last_trash_angle = 0
last_recycle_angle = 0

program_start_time = time.time()

### CENTER BOX ###
box_size = 350
pbox_xmin = (resW // 2) - (box_size // 2)
pbox_xmax = (resW // 2) + (box_size // 2)
pbox_ymin = (resH // 2) - (box_size // 2)
pbox_ymax = (resH // 2) + (box_size // 2)

### MODEL LOAD ###
if not os.path.exists(model_path):
    print("ERROR: Model path invalid.")
    sys.exit()

model = YOLO(model_path, task='detect')
labels = model.names

### CAMERA SETUP ###
if 'usb' in cam_source:
    cam = cv2.VideoCapture(int(cam_source[3:]))
    cam.set(3, resW)
    cam.set(4, resH)
    cam_type = 'usb'
else:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)}
    ))
    cam.start()
    cam_type = 'picamera'

### STATE ###
consecutive_trash = 0
consecutive_recycle = 0

trash_state = "CLOSED"
recycle_state = "CLOSED"

trash_timer = 0
recycle_timer = 0

last_action_time = 0

avg_frame_rate = 0
frame_rate_buffer = []

### MAIN LOOP ###
while True:
    t_start = time.perf_counter()
    current_time = time.time()

    # CAMERA
    if cam_type == 'usb':
        ret, frame = cam.read()
        if not ret:
            break
    else:
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # DETECTION
    results = model.track(frame, verbose=False)
    detections = results[0].boxes

    trash_locations = []
    recycle_locations = []

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

        classidx = int(det.cls.item())
        classname = labels[classidx].lower()
        conf = det.conf.item()

        if "trash" in classname:
            if conf > tra_min_thresh:
                color = (0, 0, 255)
                trash_locations.append((cx, cy))
            else:
                continue

        elif "recycle" in classname:
            if conf > rec_min_thresh:
                color = (0, 255, 0)
                recycle_locations.append((cx, cy))
            else:
                continue
        else:
            continue

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"{classname}: {int(conf*100)}%",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # CENTER CHECK
    trash_in_box = any(pbox_xmin < x < pbox_xmax and pbox_ymin < y < pbox_ymax for x, y in trash_locations)
    recycle_in_box = any(pbox_xmin < x < pbox_xmax and pbox_ymin < y < pbox_ymax for x, y in recycle_locations)

    # DEBOUNCE
    consecutive_trash = min(8, consecutive_trash + 1) if trash_in_box else max(0, consecutive_trash - 1)
    consecutive_recycle = min(8, consecutive_recycle + 1) if recycle_in_box else max(0, consecutive_recycle - 1)

    ### SERVO CONTROL ###

    # --- TRASH OPEN ---
    if (consecutive_trash >= 8 and
        trash_state == "CLOSED" and
        recycle_state == "CLOSED" and
        (current_time - last_action_time > COOLDOWN) and
        (current_time - program_start_time > STARTUP_DELAY)):

        print("Trash detected -> Opening")

        if last_trash_angle != -90:
            move_servo(trash_servo, -90)
            last_trash_angle = -90

        trash_timer = current_time
        last_action_time = current_time
        trash_state = "OPEN"

    # --- RECYCLE OPEN ---
    elif (consecutive_recycle >= 8 and
          recycle_state == "CLOSED" and
          trash_state == "CLOSED" and
          (current_time - last_action_time > COOLDOWN) and
          (current_time - program_start_time > STARTUP_DELAY)):

        print("Recycle detected -> Opening")

        if last_recycle_angle != 90:
            move_servo(recycle_servo, 90)
            last_recycle_angle = 90

        recycle_timer = current_time
        last_action_time = current_time
        recycle_state = "OPEN"

    # --- TRASH CLOSE ---
    if trash_state == "OPEN" and (current_time - trash_timer >= STAY_OPEN):
        if last_trash_angle != 0:
            move_servo(trash_servo, 0)
            last_trash_angle = 0

        time.sleep(0.4)
        trash_servo.detach()
        trash_state = "CLOSED"

    # --- RECYCLE CLOSE ---
    if recycle_state == "OPEN" and (current_time - recycle_timer >= STAY_OPEN):
        if last_recycle_angle != 0:
            move_servo(recycle_servo, 0)
            last_recycle_angle = 0

        time.sleep(0.4)
        recycle_servo.detach()
        recycle_state = "CLOSED"

    ### UI ###
    cv2.rectangle(frame, (pbox_xmin, pbox_ymin), (pbox_xmax, pbox_ymax), (0, 255, 255), 2)

    status = f"Trash: {trash_state} | Recycle: {recycle_state}"
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Smart Bin Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > 30:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

### CLEANUP ###
if cam_type == 'usb':
    cam.release()
else:
    cam.stop()

cv2.destroyAllWindows()