import os
import cv2
import time
from datetime import datetime
from dotenv import load_dotenv
import geocoder
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Optional beep
try:
    import winsound
    def play_alert_sound():
        winsound.Beep(1000, 300)
except Exception:
    import sys
    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
        def play_alert_sound():
            print("\a", end="", flush=True)
    else:
        def play_alert_sound():
            pass

# MediaPipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

load_dotenv()

# Twilio config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
RECIPIENT_PHONE_NUMBER = os.getenv("RECIPIENT_PHONE_NUMBER")

# Video source
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

# Control params
SMS_COOLDOWN = float(os.getenv("SMS_COOLDOWN", "30"))  # seconds
MAX_HANDS = int(os.getenv("MAX_HANDS", "2"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "screenshots")
LOG_FILE = os.getenv("LOG_FILE", "alert_log.txt")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    print("[WARN] Twilio credentials missing — SMS will not be sent.")

# ----------------- Location Fetching -----------------
def get_location():
    """Get approximate location via IP, fallback to Hyderabad."""
    try:
        g = geocoder.ip("me")
        if g.ok:
            lat, lon = g.latlng
            city = g.city or "Unknown"
            print(f"[INFO] Location: {city} ({lat}, {lon})")
            return lat, lon, city
    except Exception as e:
        print(f"[ERROR] Location fetch failed: {e}")
    # fallback
    lat, lon, city = 17.3918, 78.4752, "Hyderabad"
    print(f"[INFO] Using fallback: {city} ({lat}, {lon})")
    return lat, lon, city

# ----------------- Finger Detection -----------------
def fingers_up(hand_landmarks):
    """Return list [Thumb, Index, Middle, Ring, Pinky] -> 1 if up, 0 if folded"""
    fingers = []

    # Thumb (x-axis check)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (y-axis check)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# ----------------- Gesture Classification -----------------
def classify_gesture(fingers):
    """Classify based on finger states"""
    thumb, index, middle, ring, pinky = fingers

    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "Kidnap Alert"
    elif index == 1 and middle == 1 and ring == 1 and thumb == 0 and pinky == 0:
        return "Medical Emergency"
    elif thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "Distress / SOS"
    elif index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return "Testing / V-Sign"
    elif thumb == 1 and pinky == 1 and index == 0 and middle == 0 and ring == 0:
        return "Call Police"
    else:
        return None

# ----------------- SMS Sending -----------------
from twilio.base.exceptions import TwilioRestException

def send_alert(screenshot_path, gesture_name):
    """Send alert via SMS, fallback to WhatsApp if SMS blocked (30004)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lat, lon, city = get_location()
    location_link = f"https://www.google.com/maps?q={lat},{lon}"
    body = (
        f"🚨 {gesture_name} at {city}. Map: https://maps.google.com/?q={lat},{lon}"
    )
    success = False
    method_used = "SMS"

    if client:
        try:
            # --- Try SMS first ---
            message = client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=RECIPIENT_PHONE_NUMBER
            )
            status = client.messages(message.sid).fetch().status
            print(f"[Twilio SMS] SID: {message.sid} | Status: {status}")
            if status in ("sent", "delivered", "queued"):
                success = True

        except TwilioRestException as e:
            if e.code == 30004:
                print("⚠️ SMS blocked (30004). Switching to WhatsApp...")
                method_used = "WhatsApp"
                try:
                    whatsapp_msg = client.messages.create(
                        body=body,
                        from_="whatsapp:+14155238886",   # Twilio WhatsApp sandbox
                        to="whatsapp:" + RECIPIENT_PHONE_NUMBER
                    )
                    print(f"[Twilio WhatsApp] SID: {whatsapp_msg.sid} | Status: {whatsapp_msg.status}")
                    success = True
                except Exception as we:
                    print(f"[ERROR] WhatsApp sending failed: {we}")
            else:
                print(f"[ERROR] Twilio SMS failed: {e}")
    else:
        print("[WARN] Twilio not configured; alerts skipped.")

    # Play local alarm
    play_alert_sound()

    # Log to file
    with open(LOG_FILE, "a") as f:
        f.write(
            f"[{timestamp}] {gesture_name} | {city} | "
            f"{method_used}: {success} | Screenshot: {screenshot_path}\n"
        )

# ----------------- Video Capture -----------------
def open_video_source(source):
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)

# ----------------- Main Loop -----------------
def main():
    cap = open_video_source(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {VIDEO_SOURCE}")
        return

    last_sms_time = 0.0
    active_gesture = None
    total_alerts = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_detected = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    fingers = fingers_up(hand_landmarks)
                    gesture = classify_gesture(fingers)

                    if gesture:
                        gesture_detected = gesture
                        cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, (0, 0, 255), 3)
                        break

            now = time.time()
            if gesture_detected and (gesture_detected != active_gesture) and (now - last_sms_time) > SMS_COOLDOWN:
                active_gesture = gesture_detected
                last_sms_time = now
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = os.path.join(OUTPUT_DIR, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                total_alerts += 1
                print(f"[ALERT #{total_alerts}] {gesture_detected} detected. Screenshot saved.")
                send_alert(screenshot_path, gesture_detected)

            if not gesture_detected:
                active_gesture = None

            cv2.putText(frame, f"Alerts: {total_alerts}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Gesture Recognition & Alerts", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
