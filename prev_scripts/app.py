# app.py  ────────────────────────────────────────────────────────────
import base64, math, pathlib

import cv2, numpy as np, onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS         # <-- NEW

# ── model -----------------------------------------------------------
MODEL_PATH = pathlib.Path("jaundice_mobilenetv3.onnx")      # adjust if renamed
sess   = ort.InferenceSession(MODEL_PATH.as_posix(),
                              providers=["CPUExecutionProvider"])
inp    = sess.get_inputs()[0].name
IMG = 224
MEAN = np.array([0.485,0.456,0.406], np.float32).reshape(1,3,1,1)
STD  = np.array([0.229,0.224,0.225], np.float32).reshape(1,3,1,1)

def preprocess(bgr):
    h,w = bgr.shape[:2]
    s = IMG/min(h,w)
    r = cv2.resize(bgr, (int(w*s), int(h*s)))
    y0 = (r.shape[0]-IMG)//2; x0 = (r.shape[1]-IMG)//2
    crop = cv2.cvtColor(r[y0:y0+IMG, x0:x0+IMG], cv2.COLOR_BGR2RGB)
    x = crop.astype(np.float32)/255.0
    x = (x.transpose(2,0,1)[None]-MEAN)/STD
    return x.astype(np.float32)

def prob(bgr):
    logits = sess.run(None, {inp: preprocess(bgr)})[0]
    return 1/(1+math.exp(-float(logits[0,0])))

# ── Flask -----------------------------------------------------------
app = Flask(__name__, template_folder='templates')
CORS(app)                             # <-- ONE-LINE FIX

@app.route("/")
def home(): return render_template("index.html")

@app.route("/predict", methods=["POST"])   # OPTIONS now auto-allowed
def predict():
    if "file" in request.files:                          # upload
        img = cv2.imdecode(np.frombuffer(request.files["file"].read(),np.uint8),
                           cv2.IMREAD_COLOR)
    else:                                                # webcam JSON
        data_url = request.get_json(force=True)["data"]
        _, b64 = data_url.split(",",1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64),np.uint8),
                           cv2.IMREAD_COLOR)

    if img is None:
        return jsonify(error="decode_failed"), 400

    p = prob(img)
    return jsonify(label="JAUNDICE" if p>=0.5 else "normal",
                   prob=round(p,3))

if __name__ == "__main__":
    app.run(debug=True)
