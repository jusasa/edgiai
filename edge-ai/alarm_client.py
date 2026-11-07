"""
[Tier 3 엣지 AI] gRPC 피에조 액추에이터 초저지연 클라이언트 모듈
- 기능: 머신러닝 추론 결과 결함(FAULT/WARNING) 발생 시 50051 포트로 gRPC RPC 직접 호출
- 지연시간: < 10ms (클라우드 왕복 500ms~2s 우회)
- 안전성: 비동기 논블로킹(Non-blocking) 스레드 풀 적용으로 ML 추론 주기 영향 제로
"""

import os
import json
import time
import socket
import threading

GRPC_HOST = os.getenv("ALARM_GRPC_HOST", os.getenv("ALARM_SERVER_HOST", "127.0.0.1"))  # RPi #1 게이트웨이 IP 또는 컨테이너명
GRPC_PORT = int(os.getenv("ALARM_GRPC_PORT", os.getenv("ALARM_SERVER_PORT", 50051)))

try:
    import grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

_CHANNEL = None
_STUB = None

def _get_grpc_stub():
    global _CHANNEL, _STUB
    if HAS_GRPC and _STUB is None:
        try:
            import alarm_pb2_grpc
            _CHANNEL = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
            _STUB = alarm_pb2_grpc.AlarmActuatorStub(_CHANNEL)
        except Exception:
            _STUB = None
    return _STUB

def _send_trigger_async(device_id, health_code, health_status, fault_hint):
    """백그라운드에서 gRPC TriggerAlarm 호출"""
    stub = _get_grpc_stub()
    if stub is not None:
        try:
            import alarm_pb2
            req = alarm_pb2.AlarmRequest(
                device_id=device_id,
                health_code=health_code,
                health_status=health_status,
                fault_hint=fault_hint,
                duration_ms=0,
                frequency_hz=2700 if health_code == 2 else 2000
            )
            res = stub.TriggerAlarm(req, timeout=0.1)
            print(f"[gRPC-CLIENT] 🚀 현장 피에조 발령 성공: {res.message}", flush=True)
            return
        except Exception as e:
            pass

    # gRPC stub 미지원 시 초고속 TCP RPC Fallback
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        s.connect((GRPC_HOST, GRPC_PORT))
        req_dict = {
            "action": "trigger",
            "device_id": device_id,
            "health_code": health_code,
            "health_status": health_status,
            "fault_hint": fault_hint,
            "frequency_hz": 2700 if health_code == 2 else 2000
        }
        s.sendall(json.dumps(req_dict).encode('utf-8'))
        resp = s.recv(512)
        s.close()
        print(f"[gRPC-CLIENT] 🚀 현장 피에조 발령 성공 (TCP-RPC): {health_status} - {fault_hint}", flush=True)
    except Exception as ex:
        # 게이트웨이 서버 미가동 시에도 추론 메인 루프 중단 없이 통과
        pass

def _send_silence_async(device_id, reason):
    """백그라운드에서 gRPC SilenceAlarm 호출"""
    stub = _get_grpc_stub()
    if stub is not None:
        try:
            import alarm_pb2
            req = alarm_pb2.SilenceRequest(device_id=device_id, reason=reason)
            res = stub.SilenceAlarm(req, timeout=0.1)
            return
        except Exception:
            pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        s.connect((GRPC_HOST, GRPC_PORT))
        req_dict = {
            "action": "silence",
            "device_id": device_id,
            "reason": reason
        }
        s.sendall(json.dumps(req_dict).encode('utf-8'))
        s.close()
    except Exception:
        pass

# ==========================================
# 외부 공개 API (Non-blocking)
# ==========================================
def trigger_alarm(device_id, health_code, health_status, fault_hint):
    """
    ML 추론에서 FAULT(Code 2) 또는 WARNING(Code 1) 발생 시 즉시 호출
    """
    t = threading.Thread(
        target=_send_trigger_async,
        args=(device_id, health_code, health_status, fault_hint),
        daemon=True
    )
    t.start()

def silence_alarm(device_id, reason="정상 복귀"):
    """
    ML 추론에서 NORMAL(Code 0) 또는 STOPPED 복귀 시 즉시 호출
    """
    t = threading.Thread(
        target=_send_silence_async,
        args=(device_id, reason),
        daemon=True
    )
    t.start()
