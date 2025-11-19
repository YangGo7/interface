"""
Flask API Server for DINO+UNet Teeth Detection
이 API를 YangGo7/interface에 추가하여 사용할 수 있습니다.
"""

import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from predict_api import TeethDetectionModel

app = Flask(__name__)
CORS(app)  # CORS 허용 (프론트엔드 연동 시 필요)

# 전역 모델 인스턴스 (서버 시작 시 한 번만 로드)
model = None

# 모델 설정
DINO_CONFIG = "configs/DINO_4scale_cls32.py"
DINO_CHECKPOINT = "weights/dino.pth"
UNET_CHECKPOINT = "weights/unet.pth"
UNET_NUM_CLASSES = 33


def init_model():
    """모델 초기화 (서버 시작 시 호출)"""
    global model
    print("=" * 60)
    print("치아 탐지 모델 초기화 중...")
    print("=" * 60)
    model = TeethDetectionModel(
        dino_config=DINO_CONFIG,
        dino_checkpoint=DINO_CHECKPOINT,
        unet_checkpoint=UNET_CHECKPOINT,
        unet_num_classes=UNET_NUM_CLASSES
    )
    print("=" * 60)
    print("모델 로드 완료! API 서버 준비됨.")
    print("=" * 60)


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


@app.route('/api/predict/dino-unet', methods=['POST'])
def predict_dino_unet():
    """
    DINO+UNet 치아 탐지 API

    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: image (file)

    또는:
        - Content-Type: application/json
        - Body: { "image": "base64_encoded_image_string" }

    Response:
        {
            "success": true,
            "data": {
                "teeth": [
                    {
                        "fdi": "11",
                        "universal": 8,
                        "bbox": [x1, y1, x2, y2],
                        "center": [x, y],
                        "source": "DINO" | "UNET",
                        "confidence": 0.95
                    },
                    ...
                ],
                "upper_arch": ["18", "17", ..., "21", "22", ...],
                "lower_arch": ["48", "47", ..., "31", "32", ...],
                "total_detected": 28,
                "image_shape": [height, width]
            },
            "message": "Detection completed successfully"
        }
    """
    try:
        # 모델 로드 확인
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500

        # 이미지 가져오기
        image_array = None

        # 방법 1: multipart/form-data (파일 업로드)
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            # 이미지 읽기
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 방법 2: JSON (base64 인코딩된 이미지)
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data in JSON'
                }), 400

            # Base64 디코딩
            image_b64 = data['image']
            if ',' in image_b64:  # data:image/jpeg;base64, 제거
                image_b64 = image_b64.split(',')[1]

            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        else:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. Use multipart/form-data or JSON with base64 image'
            }), 400

        # 이미지 유효성 검사
        if image_array is None or image_array.size == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400

        # 예측 실행
        print(f"[API] 이미지 크기: {image_array.shape}")
        result = model.predict(image_array)

        # 응답 생성
        response = {
            'success': True,
            'data': result,
            'message': 'Detection completed successfully'
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Internal server error'
        }), 500


@app.route('/api/predict/dino-unet/annotated', methods=['POST'])
def predict_dino_unet_annotated():
    """
    DINO+UNet 치아 탐지 + 어노테이션된 이미지 반환

    Response:
        {
            "success": true,
            "data": { ... },
            "annotated_image": "base64_encoded_image_string"
        }
    """
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        # 이미지 가져오기 (이전과 동일)
        image_array = None
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.is_json:
            data = request.get_json()
            image_b64 = data['image']
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_array is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400

        # 예측 실행
        result = model.predict(image_array)

        # 이미지에 어노테이션 추가
        annotated_image = image_array.copy()
        for tooth in result['teeth']:
            x1, y1, x2, y2 = [int(v) for v in tooth['bbox']]
            fdi = tooth['fdi']
            source = tooth['source']

            # 박스 색상 (DINO: 파란색, UNET: 초록색)
            color = (255, 0, 0) if source == 'DINO' else (0, 255, 0)

            # 박스 그리기
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # FDI 번호 텍스트
            cv2.putText(
                annotated_image,
                fdi,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            'success': True,
            'data': result,
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}",
            'message': 'Detection and annotation completed successfully'
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # 모델 초기화
    init_model()

    # 서버 실행
    print("\n" + "=" * 60)
    print("Flask API Server Starting...")
    print("Endpoints:")
    print("  - GET  /health")
    print("  - POST /api/predict/dino-unet")
    print("  - POST /api/predict/dino-unet/annotated")
    print("=" * 60 + "\n")

    app.run(
        host='0.0.0.0',  # 외부 접속 허용
        port=5001,       # 포트 (기존 interface가 5000이면 다른 포트 사용)
        debug=True       # 개발 모드
    )
