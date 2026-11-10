import json
import os
import time
from collections import deque
import joblib
import numpy as np
import paho.mqtt.client as mqtt

# gRPC 초저지연 현장 피에조 부저 액추에이터 클라이언트
try:
    import alarm_client
except ImportError:
    alarm_client = None

# 게이트웨이 원격 접속 환경변수
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.50.194")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")
MODEL_DIR   = os.getenv("MODEL_DIR", os.path.dirname(__file__))

# 공유 속성 동기화 설정 및 물리 임계 안전망 기준값
CFG = {
    "fanSpeed": float(os.getenv("FAN_SPEED", "1")),
    "rmsLimit": float(os.getenv("RMS_LIMIT", "0.35")),      # 진동 가속도 RMS 한계 (g)
    "warnTh": float(os.getenv("WARN_TH", "0.60")),          # 이상 확률 경고 임계값
    "faultTh": float(os.getenv("FAULT_TH", "0.85")),        # 이상 확률 결함 임계값
    "r1x_fault": float(os.getenv("R1X_FAULT", "0.65")),     # 1x 상대 진폭 한계 (불균형)
    "kurt_fault": float(os.getenv("KURT_FAULT", "5.0")),    # 첨도 한계 (충격성 베어링 마모)
    "vel_fault": float(os.getenv("VEL_FAULT", "4.5"))       # ISO 10816-3 진동 속도 위험 한계 (mm/s)
}

# 잔여 수명(RUL) 계산용 시계열 큐 (최근 150개 = 약 5분)
HIST_SCORE = deque(maxlen=150)
HIST_RMS   = deque(maxlen=150)

# 머신러닝 모델 캐시 (바람 세기별 모델 또는 단일 모델)
MODEL_CACHE = {}

def get_model(speed):
    """
    바람 세기별 전용 모델(model_s{speed}.joblib) 우선 로드,
    부재 시 범용 모델(model.joblib) 로드 및 캐싱
    """
    speed_int = int(speed)
    if speed_int in MODEL_CACHE:
        return MODEL_CACHE[speed_int]

    speed_model_path = os.path.join(MODEL_DIR, f"model_s{speed_int}.joblib")
    default_model_path = os.path.join(MODEL_DIR, "model.joblib")

    target_path = speed_model_path if os.path.exists(speed_model_path) else default_model_path

    if os.path.exists(target_path):
        try:
            bundle = joblib.load(target_path)
            MODEL_CACHE[speed_int] = bundle
            print(f"[EDGE-AI] 머신러닝 모델 로드 완료 (속도 {speed_int}단): {target_path}", flush=True)
            if "label_encoder" in bundle:
                print(f"[EDGE-AI] 지원 클래스: {bundle['label_encoder'].classes_}", flush=True)
            return bundle
        except Exception as e:
            print(f"[EDGE-AI] 모델 로드 실패 ({target_path}): {e}", flush=True)
            return None
    else:
        return None

def predict_rul_minutes(hist_queue, target_th):
    """최근 시계열 큐를 이용한 임계 도달 시간(RUL) 1차 외삽 (Time-to-Threshold)"""
    if len(hist_queue) < 40:
        return None
    t = np.array([h[0] for h in hist_queue]) / 60000.0  # ms -> 분 단위
    s = np.array([h[1] for h in hist_queue])
    slope, icpt = np.polyfit(t - t[-1], s, 1)
    
    # 열화 진행 방향(기울기 양수) 확인 및 이미 임계치 초과 여부 점검
    if slope <= 0.0005 or icpt >= target_th:
        return None
    rul = float((target_th - icpt) / slope)
    return round(max(0.1, min(999.0, rul)), 1)

def on_connect(client, userdata, flags, rc):
    print(f"[*] 게이트웨이({MQTT_BROKER}:{MQTT_PORT}) 접속 완료 (rc={rc})", flush=True)
    # 게이트웨이 정제 특징량 및 ThingsBoard 설정 하향 토픽 구독
    client.subscribe([("factory/+/features", 1), ("config/+/attributes", 1)])
    print("[*] 구독 등록 완료: factory/+/features, config/+/attributes", flush=True)

def on_message(client, userdata, msg):
    try:
        topic_parts = msg.topic.split('/')
        
        # 1. ThingsBoard 설정 하향(config/<dev>/attributes) 수신 처리
        if topic_parts[0] == "config":
            payload_str = msg.payload.decode('utf-8')
            cfg_data = json.loads(payload_str)
            for k, v in cfg_data.items():
                if k in CFG:
                    CFG[k] = float(v)
            if "fanSpeed" in cfg_data:
                # 운전 조건(바람 세기) 변경 시 시계열 큐 초기화 (이전 조건과 간섭 배제)
                HIST_SCORE.clear()
                HIST_RMS.clear()
            print(f"[CFG] 공유 속성 갱신: {CFG}", flush=True)
            return

        # 2. 게이트웨이 정제 특징량(factory/<dev>/features) 수신
        device_name = topic_parts[1]
        data = json.loads(msg.payload.decode('utf-8'))
        current_speed = int(data.get("fan_speed", CFG["fanSpeed"]))

        acc_rms = float(data.get("acc_rms_g", 0.0))
        cf      = float(data.get("crest_factor", 0.0))
        kurt    = float(data.get("kurtosis", 0.0))
        rpm     = float(data.get("rpm_est", 0.0))
        b_1x    = float(data.get("band_1x", 0.0))
        b_harm  = float(data.get("band_harm", 0.0))
        r1x     = float(data.get("r1x", 0.0))
        rh      = float(data.get("rh", 0.0))
        vel_rms = float(data.get("vel_rms_mms", 0.0))

        feat_vector = np.array([[acc_rms, cf, kurt, rpm, b_1x, b_harm, r1x, rh]])

        # [물리 안전망 0단계: 선풍기 정지(STOPPED) 최우선 물리 판정]
        # 실측 데이터 분석 결과 off 상태의 RMS는 평균 0.0021g(열잡음) 수준임.
        # 회전수가 0이거나 RMS가 0.015g 미만인 경우 기계 결함이 아닌 '전원 OFF'로 최우선 확정
        is_physically_stopped = (acc_rms < 0.015) or (rpm == 0.0 and acc_rms < 0.03)

        # 3. 세기별 머신러닝 추론 (XGBoost)
        bundle = get_model(current_speed)
        ml_model = bundle.get("model") if bundle else None
        label_encoder = bundle.get("label_encoder") if bundle else None

        if is_physically_stopped:
            health_code = 0
            health_status = "STOPPED"
            fault_hint = "선풍기 정지"
            anomaly_score = 0.0
            # 정지 시 시계열 큐 초기화 (이전 가동 시 데이터와의 오차 및 RUL 왜곡 차단)
            HIST_SCORE.clear()
            HIST_RMS.clear()
        elif ml_model is not None and label_encoder is not None:
            pred_idx = ml_model.predict(feat_vector)[0]
            pred_label = str(label_encoder.inverse_transform([pred_idx])[0])
            proba = ml_model.predict_proba(feat_vector)[0]
            prob_dict = dict(zip(label_encoder.classes_, proba))
            prob_normal = prob_dict.get("normal", 0.0)

            # 정상 확률의 여사건을 이상 점수로 정의 (0.0 ~ 1.0 정규화)
            anomaly_score = round(float(1.0 - prob_normal), 3)

            # 기본 라벨 기반 건전성 상태 매핑
            if pred_label == "normal":
                health_code = 0
                health_status = "NORMAL"
                fault_hint = "정상 가동"
            elif pred_label == "off":
                health_code = 0
                health_status = "STOPPED"
                fault_hint = "선풍기 정지"
                anomaly_score = 0.0
                HIST_SCORE.clear()
                HIST_RMS.clear()
            elif pred_label == "unbal1g":
                health_code = 1
                health_status = "WARNING"
                fault_hint = "질량 불균형(Unbalance 1x)"
            elif pred_label == "loose":
                health_code = 2
                health_status = "FAULT"
                fault_hint = "기계적 헐거움(Looseness)"
            else:
                health_code = 1
                health_status = "WARNING"
                fault_hint = f"이상 감지 ({pred_label})"
        else:
            # 모델 미탑재 시 휴리스틱 룰 모드
            anomaly_score = 0.0
            health_code = 0
            health_status = "NORMAL"
            fault_hint = "규칙 기반 기본 모드"

        # 4. 하이브리드 물리 임계치 안전망 (Safety Net Rule)
        # ML 모델이 미처 학습하지 못한 고장이나 오분류를 방어하기 위한 다중 안전 규칙
        if health_status != "STOPPED":
            if (r1x >= CFG["r1x_fault"] and acc_rms >= CFG["rmsLimit"]) or \
               (kurt >= CFG["kurt_fault"]) or \
               (vel_rms >= CFG["vel_fault"]):
                health_code = 2
                health_status = "FAULT"
                fault_hint += " [물리 안전망: 위험 임계 초과]"
            elif anomaly_score >= CFG["faultTh"] or acc_rms >= CFG["rmsLimit"]:
                if health_code < 2:
                    health_code = 2
                    health_status = "FAULT"
                    fault_hint += " [확률/RMS 결함 임계 초과]"
            elif anomaly_score >= CFG["warnTh"]:
                if health_code < 1:
                    health_code = 1
                    health_status = "WARNING"
                    fault_hint += " [주의 임계 초과]"

        # 5. 잔여 수명(RUL) 계산 (정지 상태가 아닐 때만 큐 누적 및 외삽)
        ts = int(data.get("ts", time.time() * 1000))
        if health_status != "STOPPED":
            HIST_SCORE.append((ts, anomaly_score))
            HIST_RMS.append((ts, acc_rms))
            # 점수 기반 외삽과 RMS 기반 외삽 중 신뢰도 높은 RUL 도출
            rul_score = predict_rul_minutes(HIST_SCORE, CFG["faultTh"])
            rul_rms   = predict_rul_minutes(HIST_RMS, CFG["rmsLimit"])
            rul_val = rul_score if rul_score is not None else rul_rms
        else:
            rul_val = None

        # 6. ThingsBoard 최종 진단 텔레메트리 페이로드 생성
        out_payload = {
            **data,
            "anomaly_score": anomaly_score,
            "health_code": health_code,
            "health_status": health_status,
            "fault_hint": fault_hint
        }
        if rul_val is not None:
            out_payload["rul_min"] = rul_val

        # 게이트웨이가 구독할 텔레메트리 토픽으로 발행
        out_topic = f"factory/{device_name}/diagnostics"
        client.publish(out_topic, json.dumps(out_payload), qos=1)

        # 7. [과제 지시서 PDF 표준] ESP32 피에조 부저 RPC 명령 발행 (GPIO 25 제어)
        # 토픽: cmd/<dev>/rpc/setBuzzer, 페이로드: {"on": true/false}
        # - Critical(FAULT) 감지 시: tone(PIN_BUZZ, 2000) 2,000Hz 비상 경보음 발진
        # - NORMAL/STOPPED 복구 시: noTone(PIN_BUZZ) 즉시 묵음
        buzzer_topic = f"cmd/{device_name}/rpc/setBuzzer"
        is_buzzer_on = (health_code == 2)  # FAULT(고장) 발생 시 부저 발진
        client.publish(buzzer_topic, json.dumps({"on": is_buzzer_on}), qos=1)

        # 8. [초저지연 엣지 백업 파이프라인] gRPC 기반 피에조 액추에이션 연동
        if alarm_client is not None:
            if health_code >= 1:
                alarm_client.trigger_alarm(device_name, health_code, health_status, fault_hint)
            else:
                alarm_client.silence_alarm(device_name, reason=f"정상 상태 유지 ({health_status})")

        print(
            f"[ML] seq={data.get('seq')} | {health_status} ({fault_hint}) | "
            f"Score={anomaly_score:.3f} | RMS={acc_rms:.4f}g | "
            f"RPM={rpm:.1f} | RUL={rul_val}",
            flush=True
        )

    except Exception as e:
        print(f"[ERROR] 추론 처리 실패: {e}", flush=True)

if __name__ == "__main__":
    client = mqtt.Client(client_id="edge-ai-worker")
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(1, 30)

    print(f"[*] 게이트웨이 브로커({MQTT_BROKER}:{MQTT_PORT}) 접속 대기 중...", flush=True)
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            print(f"[*] 접속 실패 ({e}), 2초 후 재시도...", flush=True)
            time.sleep(2)

    client.loop_forever()
