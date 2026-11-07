import json
import os
import time
from collections import deque
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from scipy.stats import kurtosis

# 게이트웨이 원격 접속 환경변수
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.0.50")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER   = os.getenv("MQTT_USER", "202345021")
MQTT_PASS   = os.getenv("MQTT_PASS", "1234")

# 공유 속성 동기화 설정 (기본값)
CFG = {
    "fanSpeed": float(os.getenv("FAN_SPEED", "1")),
    "rmsLimit": 0.35,
    "warnTh": 0.60,
    "faultTh": 0.85
}

# 잔여 수명(RUL) 계산용 시계열 큐 (최근 150개 = 약 5분)
HIST = deque(maxlen=150)

# 머신러닝 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
ml_model = None
label_encoder = None
feature_cols = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        ml_model = bundle["model"]
        label_encoder = bundle["label_encoder"]
        feature_cols = bundle["feature_cols"]
        print(f"[EDGE-AI] XGBoost 모델 로드 완료: {MODEL_PATH}")
        print(f"[EDGE-AI] 지원 클래스: {label_encoder.classes_}")
    except Exception as e:
        print(f"[EDGE-AI] 모델 로드 실패: {e}")
else:
    print("[EDGE-AI] 경고: model.joblib이 없어 임시 규칙 모드로 동작합니다.")

def predict_rul_minutes(target_th):
    """최근 이상 점수 큐를 이용한 임계 도달 시간(RUL) 1차 외삽"""
    if len(HIST) < 40:
        return None
    t = np.array([h[0] for h in HIST]) / 60000.0  # 분 단위 변환
    s = np.array([h[1] for h in HIST])
    slope, icpt = np.polyfit(t - t[-1], s, 1)
    if slope <= 0.001 or icpt >= target_th:
        return None
    rul = float((target_th - icpt) / slope)
    return round(max(0.1, rul), 1)

# 엣지 AI 파이(Pi #2)의 수신부
def on_connect(client, userdata, flags, rc):
    # 게이트웨이가 넘겨주는 특징량 토픽 구독
    client.subscribe("factory/+/features", qos=1)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode('utf-8'))
    device_name = msg.topic.split('/')[1]

    # 게이트웨이가 이미 계산 완료한 특징량 바로 입력
    feat_vector = np.array([[
        data["acc_rms_g"], data["crest_factor"], data["kurtosis"],
        data["rpm_est"], data["band_1x"], data["band_harm"],
        data["r1x"], data["rh"]
    ]])

    # ML 모델 추론
    pred_idx = ml_model.predict(feat_vector)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    anomaly_score = round(float(1.0 - dict(zip(label_encoder.classes_, ml_model.predict_proba(feat_vector)[0])).get("normal", 0.0)), 3)

    # ThingsBoard Gateway가 수집할 최종 진단 토픽으로 전달
    out_payload = {
        **data,
        "anomaly_score": anomaly_score,
        "health_code": 0 if pred_label == "normal" else (1 if pred_label == "unbal1g" else 2),
        "health_status": "NORMAL" if pred_label == "normal" else ("WARNING" if pred_label == "unbal1g" else "ALARM"),
        "fault_hint": pred_label
    }
    client.publish(f"factory/{device_name}/diagnostics", json.dumps(out_payload), qos=1)

if __name__ == "__main__":
    client = mqtt.Client(client_id="edge-ai-worker")
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[*] 게이트웨이 브로커({MQTT_BROKER}:{MQTT_PORT}) 접속 대기 중...")
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception:
            time.sleep(2)

    client.loop_forever()
