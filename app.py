# from ultralytics import YOLO
# import cv2
# model = YOLO(r"D:\Guvi projects\BikeHelmetNumplate\bikemodel.pt")  # load a pretrained model (recommended for training)
# # print(model)
# result = model.predict("biketrafic.jpg")

# camera
# model = YOLO("best.pt")
# result = model.track(source=0, show=True)
# print(result)


#youtube
# from pytubefix import YouTube
# yt = YouTube("https://youtube.com/shorts/A32mPp3qb0g?si=AOvKY2RO_Vpk6OJS")
# stream = yt.streams.filter(file_extension="mp4", resolution="720p").first()
# video_path = stream.download(filename="biketraficvideo4.mp4")
# print(f"Video downloaded to: {video_path}")

# model = YOLO("bikemodel.pt")
# model.track(source="biketraficvideo4.mp4", show=True)

import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
import csv
from PIL import Image
import tempfile
import re

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "bikemodel.pt"
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.3
HELMET_CONF_THRESHOLD = 0.2
HELMET_ASSOCIATION_RADIUS = 200
HELMET_IOU_THRESHOLD = 0.3  # NEW: Minimum IoU between helmet and bike/head
VIOLATION_CONFIRM_FRAMES = 5
SAVE_VIOLATIONS = True
OUTPUT_DIR = "violations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# INIT MODELS & TOOLS (CACHED)
# ======================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'], gpu=True)

model = load_model()
reader = load_easyocr()

# Simple Tracker Class
class SimpleTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row, col] > 100: continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

# Initialize trackers
if 'trackers' not in st.session_state:
    st.session_state.bike_tracker = SimpleTracker()
    st.session_state.helmet_tracker = SimpleTracker()
    st.session_state.plate_tracker = SimpleTracker()
    st.session_state.bike_violation_state = defaultdict(lambda: {
        'no_helmet_frames': 0,
        'last_plate_text': '',
        'plate_img': None
    })
    st.session_state.frame_idx = 0

# CSV Logging
CSV_FILE = os.path.join(OUTPUT_DIR, "violations_log.csv")
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Plate Number", "Source", "Violation Confirmed"])

# ======================
# HELPER: CALCULATE IOU
# ======================
def calculate_iou(boxA, boxB):
    # Box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ======================
# OCR FUNCTION (ENHANCED)
# ======================
def validate_indian_plate(text):
    patterns = [
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$',
        r'^[A-Z]{2}[A-Z]?\d{4}[A-Z]{1,2}$',
    ]
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    return False

def ocr_plate(plate_img):
    if plate_img.size == 0:
        return ""

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 50:
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    results = reader.readtext(
        thresh,
        detail=0,
        paragraph=False,
        min_size=10,
        text_threshold=0.5,
        low_text=0.3,
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        blocklist='!@#$%^&*()_+-=[]{}|;:,.<>?/\\'
    )

    if results:
        text = "".join(results).replace(" ", "").upper()
        text = ''.join(ch for ch in text if ch.isalnum())
        fixes = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'l': '1', 'o': '0', 'i': '1'}
        fixed_text = ''.join(fixes.get(c, c) for c in text)

        if validate_indian_plate(fixed_text):
            return fixed_text
        elif validate_indian_plate(text):
            return text
        elif len(text) >= 4:
            return text

    return ""

# ======================
# PROCESS FRAME — UPDATED LOGIC
# ======================
def process_frame(frame, is_image=False):
    results = model(frame, conf=0.1, iou=IOU_THRESHOLD, augment=True)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    bikes = []
    helmets = []
    plates = []

    for i, cls in enumerate(classes):
        x1, y1, x2, y2 = boxes[i]
        conf = confs[i]

        if cls == 2:
            if conf < HELMET_CONF_THRESHOLD:
                continue
        else:
            if conf < CONF_THRESHOLD:
                continue

        if cls == 0:
            bikes.append([x1, y1, x2, y2])
        elif cls == 1:
            plates.append([x1, y1, x2, y2])
        elif cls == 2:
            helmets.append([x1, y1, x2, y2])

    st.session_state.last_detection_stats = {
        'bikes': len(bikes),
        'helmets': len(helmets),
        'plates': len(plates),
        'raw_detections': len(classes),
        'helmet_confs': [float(round(confs[i], 2)) for i, c in enumerate(classes) if c == 2]
    }

    if is_image:
        violations = []

        for bike_box in bikes:
            bx1, by1, bx2, by2 = bike_box

            # 🔥 NEW LOGIC: Check IoU between helmet and bike
            helmet_on_rider = False
            best_iou = 0.0
            for helmet_box in helmets:
                iou = calculate_iou(helmet_box, bike_box)
                if iou > best_iou:
                    best_iou = iou
                if iou >= HELMET_IOU_THRESHOLD:
                    helmet_on_rider = True
                    break

            if not helmet_on_rider and plates:
                best_plate = None
                min_dist = float('inf')
                for plate_box in plates:
                    px1, py1, px2, py2 = plate_box
                    plate_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                    bike_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                    dist = np.linalg.norm(np.array(bike_center) - np.array(plate_center))
                    if dist < min_dist:
                        min_dist = dist
                        best_plate = plate_box

                if best_plate is not None:
                    px1, py1, px2, py2 = map(int, best_plate)
                    plate_img = frame[py1:py2, px1:px2].copy()
                    plate_text = ocr_plate(plate_img)

                    if plate_text and len(plate_text) >= 4:
                        violations.append({
                            'bike_box': bike_box,
                            'plate_box': best_plate,
                            'plate_text': plate_text,
                            'plate_img': plate_img,
                        })

                        if SAVE_VIOLATIONS:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(OUTPUT_DIR, f"violation_{timestamp}_image.jpg")
                            cv2.imwrite(img_path, frame)
                            plate_img_path = os.path.join(OUTPUT_DIR, f"plate_{timestamp}_image.jpg")
                            cv2.imwrite(plate_img_path, plate_img)

                            with open(CSV_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([timestamp, plate_text, "IMAGE", "YES"])

                        st.sidebar.warning(f"🚨 VIOLATION (Image): {plate_text}")

            bx1, by1, bx2, by2 = map(int, bike_box)
            color = (0, 0, 255) if not helmet_on_rider else (0, 255, 0)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            label = f"NO HELMET! (IoU:{best_iou:.2f})" if not helmet_on_rider else f"Helmet OK (IoU:{best_iou:.2f})"
            cv2.putText(frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for plate_box in plates:
            px1, py1, px2, py2 = map(int, plate_box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, "Plate", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for helmet_box in helmets:
            hx1, hy1, hx2, hy2 = map(int, helmet_box)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
            cv2.putText(frame, "Helmet", (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for v in violations:
            px1, py1, px2, py2 = map(int, v['plate_box'])
            cv2.putText(frame, f"PLATE: {v['plate_text']}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"B:{len(bikes)} H:{len(helmets)} P:{len(plates)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, violations

    # Video/Webcam mode
    bike_centroids = st.session_state.bike_tracker.update(bikes)
    helmet_centroids = st.session_state.helmet_tracker.update(helmets)
    plate_centroids = st.session_state.plate_tracker.update(plates)

    current_bikes = {id: box for id, box in zip(bike_centroids.keys(), bikes[:len(bike_centroids)])} if len(bikes) > 0 else {}
    current_helmets = {id: box for id, box in zip(helmet_centroids.keys(), helmets[:len(helmet_centroids)])} if len(helmets) > 0 else {}
    current_plates = {id: box for id, box in zip(plate_centroids.keys(), plates[:len(plate_centroids)])} if len(plates) > 0 else {}

    violations = []

    for bike_id, bike_box in current_bikes.items():
        bx1, by1, bx2, by2 = bike_box

        # 🔥 NEW LOGIC: Check IoU between helmet and bike
        helmet_on_rider = False
        best_iou = 0.0
        for helmet_box in current_helmets.values():
            iou = calculate_iou(helmet_box, bike_box)
            if iou > best_iou:
                best_iou = iou
            if iou >= HELMET_IOU_THRESHOLD:
                helmet_on_rider = True
                break

        if not helmet_on_rider:
            st.session_state.bike_violation_state[bike_id]['no_helmet_frames'] += 1
        else:
            st.session_state.bike_violation_state[bike_id]['no_helmet_frames'] = 0

        if st.session_state.bike_violation_state[bike_id]['no_helmet_frames'] >= VIOLATION_CONFIRM_FRAMES:
            best_plate = None
            min_dist = float('inf')
            for plate_box in current_plates.values():
                px1, py1, px2, py2 = plate_box
                plate_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                bike_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                dist = np.linalg.norm(np.array(bike_center) - np.array(plate_center))
                if dist < min_dist:
                    min_dist = dist
                    best_plate = plate_box

            if best_plate is not None:
                px1, py1, px2, py2 = map(int, best_plate)
                plate_img = frame[py1:py2, px1:px2].copy()

                if st.session_state.bike_violation_state[bike_id]['last_plate_text'] == '':
                    plate_text = ocr_plate(plate_img)
                    st.session_state.bike_violation_state[bike_id]['last_plate_text'] = plate_text
                    st.session_state.bike_violation_state[bike_id]['plate_img'] = plate_img

                    if plate_text and len(plate_text) >= 4:
                        violations.append({
                            'bike_id': bike_id,
                            'bike_box': bike_box,
                            'plate_box': best_plate,
                            'plate_text': plate_text,
                            'plate_img': plate_img
                        })

                        if SAVE_VIOLATIONS:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(OUTPUT_DIR, f"violation_{timestamp}_bike{bike_id}.jpg")
                            cv2.imwrite(img_path, frame)
                            plate_img_path = os.path.join(OUTPUT_DIR, f"plate_{timestamp}_bike{bike_id}.jpg")
                            cv2.imwrite(plate_img_path, plate_img)

                            with open(CSV_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([timestamp, plate_text, st.session_state.frame_idx, "YES"])

                        st.sidebar.warning(f"🚨 VIOLATION: Bike {bike_id} → {plate_text}")

        bx1, by1, bx2, by2 = map(int, bike_box)
        color = (0, 0, 255) if st.session_state.bike_violation_state[bike_id]['no_helmet_frames'] > 0 else (0, 255, 0)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, f"Bike {bike_id}", (bx1, by1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"NoH:{st.session_state.bike_violation_state[bike_id]['no_helmet_frames']} IoU:{best_iou:.2f}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for plate_box in current_plates.values():
        px1, py1, px2, py2 = map(int, plate_box)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

    for helmet_box in current_helmets.values():
        hx1, hy1, hx2, hy2 = map(int, helmet_box)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)

    for v in violations:
        px1, py1, px2, py2 = map(int, v['plate_box'])
        cv2.putText(frame, f"PLATE: {v['plate_text']}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"B:{len(current_bikes)} H:{len(current_helmets)} P:{len(current_plates)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, violations

# ======================
# CUSTOM CSS FOR SUPER DOOPER UI
# ======================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #e94560 !important;
        font-weight: 700;
    }
    .stButton>button {
        background: #e94560;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #ff6b81;
        transform: scale(1.05);
    }
    .stMetric {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }
    .stExpander {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    .css-1v0mbdj, .css-1y4p8pa {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
    }
    .stProgress > div > div > div > div {
        background: #e94560;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #e94560;
    }
    .sidebar .sidebar-content {
        background: rgba(0,0,0,0.2);
    }
    .stAlert {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="🚦 SUPER DOOPER Helmet Violation Detector", layout="wide", page_icon="🚨")

# Dark/Light Mode Toggle
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

col_toggle, col_title = st.columns([1, 10])
with col_toggle:
    if st.button("🌓 Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

with col_title:
    st.title("🚦 SUPER DOOPER Helmet Violation Detector")
    st.caption("AI-Powered Real-Time Detection • YOLO + EasyOCR • Stop/Pause/Resume • Sleek UI")

if not st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); color: #1a1a2e; }
        h1, h2, h3 { color: #1a1a2e !important; }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎞️ Smart Video Control", "🎥 Live Cam Detection"])

# ======================
# SIDEBAR: VIOLATION LOG + STATS
# ======================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/motorcycle.png", width=50)
    st.title("📊 Control Panel")

    if 'last_detection_stats' in st.session_state:
        stats = st.session_state.last_detection_stats
        st.markdown("### 📈 Real-Time Stats")
        col1, col2 = st.columns(2)
        col1.metric("🏍️ Bikes", stats['bikes'])
        col2.metric("🛡️ Helmets", stats['helmets'])
        col1, col2 = st.columns(2)
        col1.metric("🔢 Plates", stats['plates'])
        if stats['helmet_confs']:
            conf_str = ", ".join(f"{x:.2f}" for x in stats['helmet_confs'])
            col2.markdown(f"**🎯 Helmet Conf:**<br>{conf_str}", unsafe_allow_html=True)

    st.divider()
    st.subheader("📂 Violation Log")
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            st.download_button("📥 Download Violations CSV", f, "violations_log.csv", "text/csv", key='download-csv')
        with open(CSV_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                st.write("**Recent Violations:**")
                for line in lines[-5:1:-1]:
                    st.code(line.strip())
    else:
        st.info("📭 No violations yet")

    st.divider()
    st.caption("Built with ❤️ using Streamlit • YOLOv11n • EasyOCR")
    st.divider()
    st.caption("Built by SAI SUDHARSAN ❤️")

# ======================
# IMAGE TAB
# ======================
with tab1:
    st.header("🖼️ Upload Image for Instant Analysis")
    st.markdown("Upload a traffic image to detect bikes without helmets and extract license plates instantly.")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_image:
        with st.spinner("🧠 AI is analyzing your image..."):
            image = Image.open(uploaded_image)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame, violations = process_frame(frame, is_image=True)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        st.image(processed_frame, caption="🔍 AI Processed Result", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        stats = st.session_state.last_detection_stats
        col1.metric("🏍️ Bikes", stats['bikes'])
        col2.metric("🛡️ Helmets", stats['helmets'])
        col3.metric("🔢 Plates", stats['plates'])

        if violations:
            st.success(f"🚨 Found {len(violations)} Violations!", icon="🚨")
            for i, v in enumerate(violations):
                with st.expander(f"INFRINGEMENT #{i+1}: {v['plate_text']}", expanded=True):
                    col_left, col_right = st.columns([1, 2])
                    with col_left:
                        st.image(cv2.cvtColor(v['plate_img'], cv2.COLOR_BGR2RGB), caption="📸 Original Plate", use_container_width=True)
                        
                        gray = cv2.cvtColor(v['plate_img'], cv2.COLOR_BGR2GRAY)
                        h, w = gray.shape
                        if h < 50:
                            gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
                        gray = cv2.bilateralFilter(gray, 9, 75, 75)
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        gray = cv2.filter2D(gray, -1, kernel)
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        st.image(thresh, caption="🔍 OCR Preprocessed", use_container_width=True)
                    with col_right:
                        st.markdown(f"### 🆔 Detected Plate: `{v['plate_text']}`")
                        st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            if stats['bikes'] > 0 and stats['helmets'] == 0:
                st.warning("⚠️ Bikes detected without helmets — but no readable plate found.", icon="⚠️")
            else:
                st.success("✅ All riders are following helmet rules!", icon="✅")

# ======================
# VIDEO TAB - WITH STOP/PAUSE/RESUME
# ======================
with tab2:
    st.header("🎞️ Smart Video Analyzer")
    st.markdown("Upload a video for frame-by-frame analysis. Control playback with Stop, Pause, and Resume.")
    
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video_uploader")

    # Initialize video control flags
    if 'video_controls' not in st.session_state:
        st.session_state.video_controls = {
            'stop': False,
            'pause': False,
            'total_frames': 0,
            'current_frame': 0
        }

    if uploaded_video:
        # Reset controls when new video uploaded
        st.session_state.video_controls = {
            'stop': False,
            'pause': False,
            'total_frames': 0,
            'current_frame': 0
        }

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Get total frames for progress bar
        cap_check = cv2.VideoCapture(video_path)
        st.session_state.video_controls['total_frames'] = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_check.release()

        # Control Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏹️ STOP", type="primary", use_container_width=True):
                st.session_state.video_controls['stop'] = True
                st.session_state.video_controls['pause'] = False
        with col2:
            if st.button("⏸️ PAUSE" if not st.session_state.video_controls['pause'] else "▶️ RESUME", use_container_width=True):
                st.session_state.video_controls['pause'] = not st.session_state.video_controls['pause']
        with col3:
            if st.button("🔁 RESET", use_container_width=True):
                st.session_state.video_controls = {
                    'stop': False,
                    'pause': False,
                    'total_frames': st.session_state.video_controls['total_frames'],
                    'current_frame': 0
                }
                st.rerun()

        stframe = st.empty()
        status_col, progress_col = st.columns([3, 1])
        status_text = status_col.empty()
        progress_bar = progress_col.empty()

        cap = cv2.VideoCapture(video_path)
        violation_list = []

        while cap.isOpened():
            if st.session_state.video_controls['stop']:
                st.warning("⏹️ Video processing stopped by user.")
                break

            if st.session_state.video_controls['pause']:
                status_text.info("⏸️ Paused. Press RESUME to continue.")
                st.button("▶️ RESUME", key="resume_btn", use_container_width=True)
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, violations = process_frame(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(processed_frame, channels="RGB", use_container_width=True)

            if violations:
                for v in violations:
                    if not any(vi['plate_text'] == v['plate_text'] for vi in violation_list):
                        violation_list.append(v)

            # Update progress
            st.session_state.video_controls['current_frame'] += 1
            if st.session_state.video_controls['total_frames'] > 0:
                progress = st.session_state.video_controls['current_frame'] / st.session_state.video_controls['total_frames']
                progress_bar.progress(progress, text=f"Frame {st.session_state.video_controls['current_frame']}")

            # Status
            stats = st.session_state.last_detection_stats
            status_text.markdown(f"**Frame {st.session_state.video_controls['current_frame']}** | 🏍️ Bikes: {stats['bikes']} | 🛡️ Helmets: {stats['helmets']} | 🔢 Plates: {stats['plates']}")

        cap.release()

        if violation_list:
            st.success(f"✅ Total violations found: {len(violation_list)}", icon="✅")
            for i, v in enumerate(violation_list):
                with st.expander(f"INFRINGEMENT #{i+1}: {v['plate_text']}", expanded=False):
                    st.code(f"Plate Number: {v['plate_text']}")
                    st.image(cv2.cvtColor(v['plate_img'], cv2.COLOR_BGR2RGB), caption="Extracted Plate", width=300)
        else:
            if not st.session_state.video_controls['stop']:
                st.info("✅ No violations detected in this video.", icon="✅")

# ======================
# WEBCAM TAB
# ======================
with tab3:
    st.header("🎥 Live Camera Detection")
    st.markdown("Real-time detection from your webcam. Perfect for traffic monitoring setups.")
    
    run = st.checkbox("🟢 START LIVE DETECTION", key="webcam_toggle", value=False)
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Check permissions.")
        else:
            st.info("📹 Live feed active. Uncheck the box to stop.")
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame.")
                    break

                processed_frame, violations = process_frame(frame)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(processed_frame, channels="RGB", use_container_width=True)

                st.session_state.frame_idx += 1

                if not st.session_state.get('webcam_toggle', False):
                    break

            cap.release()
            st.info("⏹️ Camera stopped successfully.")
    else:
        st.info("▶️ Check the box above to start live detection.")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("### 🚨 SUPER DOOPER Helmet Violation Detector")
st.caption("Made with ❤️ for smarter traffic enforcement • Supports Images, Videos & Live Camera • EasyOCR Enhanced • Stop/Pause Controls")