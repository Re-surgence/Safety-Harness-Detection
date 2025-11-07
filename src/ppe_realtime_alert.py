import cv2
from ultralytics import YOLO
import numpy as np
from playsound import playsound  # pip install playsound
from datetime import datetime
import os
import csv  # For summary
import os  # Add this if not already (it's there via imports)

# Base dir: Assumes script run from repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets repo root from src/

# Paths relative to repo root
model_path = os.path.join(BASE_DIR, 'models', 'best.pt')
ALERT_SOUND = os.path.join(BASE_DIR, 'data', 'beep.wav')
video_path = os.path.join(BASE_DIR, 'data', 'tocheck.mp4')

# Load model (rest unchanged)
model = YOLO(model_path)

# ... (keep the ALERT_SOUND check as-is)

# Video cap (rest unchanged)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Can't open video {video_path}")
    exit()
# Load your trained model (update path to your local best.pt)
model = YOLO(model_path)

# Your classes (from training output)
classes = {
    0: 'helmet', 1: 'gloves', 2: 'vest', 3: 'boots', 4: 'goggles',
    5: 'none', 6: 'Person', 7: 'no_helmet', 8: 'no_goggle', 9: 'no_gloves',
    10: 'no_boots'
}
person_class = 6  # 'Person'
critical_no_classes = [7, 8, 9, 10]  # no_helmet, no_goggle, no_gloves, no_boots
no_to_missing = {7: 'helmet', 8: 'goggles', 9: 'gloves', 10: 'boots'}  # Map no_ IDs to missing item names

# Alert & Save setup
os.makedirs('alert_logs', exist_ok=True)
os.makedirs('saved_frames/compliant', exist_ok=True)  # Folder for compliant saves
os.makedirs('saved_frames/non_compliant', exist_ok=True)  # Folder for non-compliant saves
alert_triggered = False
alert_debounce = 60  # Seconds between alerts (avoid spam)
save_counter = 0  # For periodic compliant saves
non_compliant_counter = 0  # Throttle non-compliant saves
alerts_data = []  # For CSV summary

def log_alert(missing_items, frame_count):
    missing_str = ', '.join(missing_items) if missing_items else 'unknown'
    with open('alert_logs/ppe_alerts.log', 'a') as f:
        f.write(f"{datetime.now()}: Non-compliance at frame {frame_count}: Missing {missing_str}\n")

def save_frame(frame, folder, prefix, frame_count, missing_items=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if missing_items:
        filename = f"{prefix}_frame_{frame_count}_{timestamp}_missing_{'_'.join(missing_items)}.jpg"
    else:
        filename = f"{prefix}_frame_{frame_count}_{timestamp}.jpg"
    filepath = os.path.join('saved_frames', folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"Saved: {filepath}")
    return filepath  # For CSV

def save_summary():
    if alerts_data:
        with open('alerts_summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'Missing Items', 'Filepath'])
            writer.writerows(alerts_data)
        print(f"📊 Summary saved: alerts_summary.csv ({len(alerts_data)} alerts)")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Can't open video {video_path}")
    exit()

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Loaded video: {total_frames} frames")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video—restarting loop...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
        continue

    frame_count += 1
    if frame_count % 10 != 0:  # Run inference every 10 frames (more frequent for better alert capture)
        cv2.imshow('PPE Safety Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run inference
    results = model(frame, verbose=False, conf=0.3)  # Adjust conf as needed

    non_compliant_detected = False
    missing_items = []  # Track specifics this frame
    persons_without_vest = []  # Temp for vest checks

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = classes[cls_id]

                if conf > 0.5:
                    color = (0, 255, 0) if cls_id not in critical_no_classes else (0, 0, 255)  # Green for good, red for no_
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Direct no_ detection: Add to missing
                    if cls_id in critical_no_classes:
                        missing_item = no_to_missing.get(cls_id, 'PPE')
                        if missing_item not in missing_items:  # Dedupe
                            missing_items.append(missing_item)
                        non_compliant_detected = True
                        cv2.putText(frame, f"Missing: {missing_item}!", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Vest/harness check per Person (simple frame-wide for now; add IoU if needed)
                    if cls_id == person_class:
                        # Check if any vest in frame (expand later)
                        has_vest = False
                        for b in boxes:
                            if classes[int(b.cls[0].cpu().numpy())] == 'vest' and b.conf[0].cpu().numpy() > 0.5:
                                has_vest = True
                                break
                        if not has_vest:
                            persons_without_vest.append((x1, y1, x2, y2))  # Track for overlay
                            if 'vest/harness' not in missing_items:
                                missing_items.append('vest/harness')
                            non_compliant_detected = True

    # Overlay missing list on first non-compliant person (if any)
    if persons_without_vest and missing_items:
        px, py, _, _ = persons_without_vest[0]  # First person
        cv2.putText(frame, f"Missing: {', '.join(missing_items)}", (int(px), int(py)+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save logic
    if non_compliant_detected:
        non_compliant_counter += 1
        # Save every Nth non-compliant (set to 1 for ALL)
        if non_compliant_counter % 10 == 0:  # Change to 2+ to throttle
            filepath = save_frame(frame, 'non_compliant', 'non_compliant', frame_count, missing_items)
            alerts_data.append([frame_count, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ', '.join(missing_items), filepath])
        
        cv2.putText(frame, f"CRITICAL: PPE NON-COMPLIANCE - {', '.join(missing_items)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Debounced alert (beep/log only once per sequence)
        if not alert_triggered:
            if os.path.exists(ALERT_SOUND):
                playsound(ALERT_SOUND)
            log_alert(missing_items, frame_count)
            alert_triggered = True
    else:
        # Reset trigger and save compliant periodically
        alert_triggered = False
        save_counter += 1
        if save_counter % 10 == 0:
            save_frame(frame, 'compliant', 'compliant', frame_count)

    # Display
    cv2.imshow('PPE Safety Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
save_summary()  # Generate CSV
print("Monitoring stopped. Check alert_logs/ for detailed records.")
print("Saved frames in saved_frames/compliant/ and saved_frames/non_compliant/")