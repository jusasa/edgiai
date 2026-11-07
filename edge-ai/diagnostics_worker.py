import json
import os
import time
from collections import deque
import joblib
import numpy as np
import paho.mqtt.client as mqtt

# 게이트웨이 원격 접속 환경변수
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.50.194")
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
        print(f"[EDGE-AI] XGBoost 모델 로드 완료: {MODEL_PATH}", flush=True)
        print(f"[EDGE-AI] 지원 클래스: {label_encoder.classes_}", flush=True)
    except Exception as e:
        print(f"[EDGE-AI] 모델 로드 실패: {e}", flush=True)
else:
    print("[EDGE-AI] 경고: model.joblib이 없어 임시 규칙 모드로 동작합니다.", flush=True)

def predict_rul_minutes(target_th):
    """최근 이상 점수 큐를 이용한 임계 도달 시간(RUL) 1차 외삽"""
    if len(HIST) < 40:
        return None
    t = np.array([h[0] for h in HIST]) / 60000.0  # ms -> 분 단위
    s = np.array([h[1] for h in HIST])
    slope, icpt = np.polyfit(t - t[-1], s, 1)
    if slope <= 0.001 or icpt >= target_th:
        return None
    rul = float((target_th - icpt) / slope)
    return round(max(0.1, rul), 1)

def on_connect(client, userdata, flags, rc):
    print(f"[*] 게이트웨이({MQTT_BROKER}:{MQTT_PORT}) 접속 완료 (rc={rc})", flush=True)
    # 게이트웨이 특징량 및 설정 하향 토픽 구독
    client.subscribe([("factory/+/features", 1), ("config/+/attributes", 1)])
    print("[*] 구독 등록 완료: factory/+/features, config/+/attributes", flush=True)

def on_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        
        # 1. ThingsBoard 설정 하향(config/machine01/attributes) 수신 처리
        if topic_parts[0] == "config":
            payload_str = msg.payload.decode('utf-8')
            cfg_data = json.loads(payload_str)
            for k, v in cfg_data.items():
                if k in CFG:
                    CFG[k] = float(v)
            if "fanSpeed" in cfg_data:
                HIST.clear()  # 운전 조건 변경 시 RUL 추세 큐 초기화
            print(f"[CFG] 공유 속성 갱신: {CFG}", flush=True)
            return

        # 2. 게이트웨이 정제 특징량(factory/machine01/features) 수신 및 ML 추론
        device_name = topic_parts[1]
        data = json.loads(msg.payload.decode('utf-8'))

        feat_vector = np.array([[
            data["acc_rms_g"], data["crest_factor"], data["kurtosis"],
            data["rpm_est"], data["band_1x"], data["band_harm"],
            data["r1x"], data["rh"]
        ]])

        if ml_model is not None:
            pred_idx = ml_model.predict(feat_vector)[0]
            pred_label = str(label_encoder.inverse_transform([pred_idx])[0])
            proba = ml_model.predict_proba(feat_vector)[0]
            prob_dict = dict(zip(label_encoder.classes_, proba))
            prob_normal = prob_dict.get("normal", 0.0)

            # 상태 및 점수 매핑 (off를 STOPPED/Code 0으로 정상 처리)
            if pred_label == "normal":
                health_code = 0
                health_status = "NORMAL"
                fault_hint = "정상 가동"
                anomaly_score = round(float(1.0 - prob_normal), 3)
            elif pred_label == "off":
                health_code = 0
                health_status = "STOPPED"
                fault_hint = "선풍기 정지"
                anomaly_score = 0.0
            elif pred_label == "unbal1g":
                health_code = 1
                health_status = "WARNING"
                fault_hint = "질량 불균형(Unbalance 1x)"
                anomaly_score = round(float(1.0 - prob_normal), 3)
            elif pred_label == "loose":
                health_code = 2
                health_status = "ALARM"
                fault_hint = "기계적 헐거움(Looseness)"
                anomaly_score = round(float(1.0 - prob_normal), 3)
            else:
                health_code = 1
                health_status = "WARNING"
                fault_hint = f"이상 감지 ({pred_label})"
                anomaly_score = round(float(1.0 - prob_normal), 3)
        else:
            health_code = 0
            health_status = "NORMAL"
            fault_hint = "임시 모드"
            anomaly_score = 0.0

        ts = int(data.get("ts", time.time() * 1000))
        if health_status != "STOPPED":
            HIST.append((ts, anomaly_score))
        rul_val = predict_rul_minutes(CFG["faultTh"])

        # ThingsBoard Gateway 최종 텔레메트리 페이로드
        out_payload = {
            **data,
            "anomaly_score": anomaly_score,
            "health_code": health_code,
            "health_status": health_status,
            "fault_hint": fault_hint
        }
        if rul_val is not None:
            out_payload["rul_min"] = rul_val

        out_topic = f"factory/{device_name}/diagnostics"
        client.publish(out_topic, json.dumps(out_payload), qos=1)

        print(
            f"[ML] seq={data.get('seq')} | {health_status} ({fault_hint}) | "
            f"Score={anomaly_score:.3f} | RMS={data.get('acc_rms_g')}g | "
            f"RPM={data.get('rpm_est')} | RUL={rul_val}",
            flush=True
        )

    except Exception as e:
        print(f"[ERROR] 추론 처리 실패: {e}", flush=True)

if __name__ == "__main__":
    client = mqtt.Client(client_id="edge-ai-worker")
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[*] 게이트웨이 브로커({MQTT_BROKER}:{MQTT_PORT}) 접속 대기 중...", flush=True)
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            print(f"[*] 접속 실패 ({e}), 2초 후 재시도...", flush=True)
            time.sleep(2)

    client.loop_forever()
