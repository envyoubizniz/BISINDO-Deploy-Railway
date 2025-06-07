import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import keras
from tensorflow.keras import layers

# ─── CONFIGURASI ────────────────────────────────────────────────────────────────
IMG_SIZE = 256

# Path relatif ke file model_ujicoba.keras di dalam folder "model/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_ujicoba.keras")

# ─── DEFINE CUSTOM LAYER SILU ───────────────────────────────────────────────────
class SiluLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return keras.activations.swish(inputs)

    def get_config(self):
        config = super().get_config()
        return config

# ─── MUAT MODEL ─────────────────────────────────────────────────────────────────
try:
    model = load_model(MODEL_PATH, custom_objects={"Silu": SiluLayer})
    print("Model berhasil dimuat dari:", MODEL_PATH)
except FileNotFoundError:
    print(f"Error: File model tidak ditemukan di {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

if model is None:
    raise RuntimeError("Gagal memuat model. Pastikan file model_ujicoba.keras ada di folder 'model/'.")

# ─── INISIALISASI MEDIAPIPE HANDS ────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ─── FUNGSI PENOLONG ─────────────────────────────────────────────────────────────
def landmarks_to_input_image(landmarks_list, img_size=IMG_SIZE, point_radius=5):
    """
    Mengonversi satu atau dua set landmark (MediaPipe) menjadi citra grayscale 256×256,
    lalu dikonversi ke BGR dan dinormalisasi ke [0,1], untuk input model.
    """
    zero_hand = np.zeros(42, dtype=np.float32)
    hand1, hand2 = zero_hand, zero_hand

    if len(landmarks_list) == 1:
        hand1 = landmarks_list[0]
    elif len(landmarks_list) >= 2:
        avg_x_0 = np.mean(landmarks_list[0][0::2])
        avg_x_1 = np.mean(landmarks_list[1][0::2])
        if avg_x_0 < avg_x_1:
            hand_left, hand_right = landmarks_list[0], landmarks_list[1]
        else:
            hand_left, hand_right = landmarks_list[1], landmarks_list[0]
        # Asumsi model dilatih: tangan kanan lalu tangan kiri
        hand1 = hand_right
        hand2 = hand_left

    # Gabungkan koordinat x,y dari kedua tangan
    x_coords = np.array(list(hand1[0::2]) + list(hand2[0::2]))
    y_coords = np.array(list(hand1[1::2]) + list(hand2[1::2]))
    x_coords = np.nan_to_num(x_coords, 0)
    y_coords = np.nan_to_num(y_coords, 0)

    valid_mask = (x_coords != 0) | (y_coords != 0)
    valid_x, valid_y = x_coords[valid_mask], y_coords[valid_mask]
    if valid_x.size > 0 and valid_y.size > 0:
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1

    x_range = max_x - min_x if (max_x - min_x) > 0 else 1e-6
    y_range = max_y - min_y if (max_y - min_y) > 0 else 1e-6

    padding = 0.1
    effective = img_size * (1 - 2 * padding)
    scale_x = effective / x_range
    scale_y = effective / y_range
    offset_x = (img_size - (max_x - min_x) * scale_x) / 2
    offset_y = (img_size - (max_y - min_y) * scale_y) / 2

    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    all_x = np.concatenate([hand1[0::2], hand2[0::2]])
    all_y = np.concatenate([hand1[1::2], hand2[1::2]])
    for i, (x, y) in enumerate(zip(all_x, all_y)):
        if valid_mask[i]:
            x_c = int((x - min_x) * scale_x + offset_x)
            y_c = int((y - min_y) * scale_y + offset_y)
            if 0 <= x_c < img_size and 0 <= y_c < img_size:
                cv2.circle(canvas, (x_c, y_c), point_radius, (255,), thickness=-1)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas_norm = canvas_bgr.astype(np.float32) / 255.0
    return canvas_norm

# ─── INISIALISASI FLASK ──────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def halaman_utama():
    """
    Render landing page: halaman_utama.html
    """
    return render_template('halaman_utama.html')

@app.route('/index')
def halaman_index():
    """
    Render page untuk akses webcam & prediksi:
    index.html
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Menerima JSON dengan field 'image' berisi base64 data JPEG.
    Mengembalikan JSON {'gesture': 'X', 'confidence': 0.95}.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Tidak ada data gambar'}), 400

    # Hapus prefix "data:image/jpeg;base64," dan decode
    img_b64 = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Gagal dekode gambar'}), 500

    # Flip horizontal agar mirror
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'gesture': 'No Hand Detected', 'confidence': 0.0})

    landmarks_list = []
    for hand_landmarks in results.multi_hand_landmarks:
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append(lm.x)
            pts.append(lm.y)
        landmarks_list.append(np.array(pts, dtype=np.float32))

    try:
        input_img = landmarks_to_input_image(landmarks_list, img_size=IMG_SIZE)
        input_tensor = np.expand_dims(input_img, axis=0)
        preds = model.predict(input_tensor, verbose=0)
        idx = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][idx])

        # Daftar label A-Z
        labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        gesture = labels[idx] if 0 <= idx < len(labels) else 'Unknown'
    except Exception as e:
        return jsonify({'gesture': 'Error', 'confidence': 0.0, 'detail': str(e)}), 500

    return jsonify({'gesture': gesture, 'confidence': confidence})

if __name__ == '__main__':
    # debug=True hanya untuk development lokal
    app.run()
