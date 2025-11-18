"""
Flask Application - Object Detection API
YOLO 기반 오브젝트 디텍션 API 서버
"""

from flask import Flask, request, jsonify , send_from_directory
from flask_cors import CORS
import os
from config import config

# Flask 앱 생성
base_dir  = os.path.abspath(os.path.dirname(__file__))
front_dir = os.path.join(base_dir , '../frontend')
app = Flask(__name__ , static_folder=front_dir , static_url_path='')
# 환경 설정 로드 (기본: development)
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# CORS 설정 (프론트엔드와 통신 가능하게)
CORS(app, resources={
    r"/api/*": {
        "origins": app.config['CORS_ORIGINS'],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})


# ==================== 기본 라우트 ====================

@app.route("/")
def index():
    "front end return(index.html)"
    return send_from_directory(app.static_folder , 'index.html')

@app.route('/api/check_server', methods=['GET'])
def check_server():
    """
    서버 상태 확인 (Health Check)
    """
    return jsonify({
        "message": "Object Detection API is running",
        "version": "1.0.0",
        "status": "healthy"
    })


@app.route('/api/health', methods=['GET'])
def health():
    """
    API 상태 확인
    """
    return jsonify({
        "status": "ok",
        "message": "API is working"
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    사용 가능한 모델 목록 반환
    """
    models = []
    for model_name, model_info in app.config['SUPPORTED_MODELS'].items():
        models.append({
            "name": model_name,
            "description": model_info["description"],
            "size": model_info["size"],
            "path": model_info["path"]
        })

    return jsonify({
        "success": True,
        "models": models,
        "default_model": app.config['DEFAULT_MODEL']
    })


@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """
    객체 탐지 API
    이미지를 받아서 YOLO 모델로 객체를 탐지합니다.
    """
    try:
        # 1. 이미지 파일 확인
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "이미지 파일이 없습니다.",
                "error_type": "NoImageFile"
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "파일이 선택되지 않았습니다.",
                "error_type": "EmptyFilename"
            }), 400

        # 2. 파일 확장자 확인
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": f"지원하지 않는 파일 형식입니다. 허용 형식: {app.config['ALLOWED_EXTENSIONS']}",
                "error_type": "InvalidFileFormat"
            }), 400

        # 3. 모델 선택
        model_name = request.form.get('model', app.config['DEFAULT_MODEL'])

        # 모델 경로 가져오기
        model_info = None
        for key, info in app.config['SUPPORTED_MODELS'].items():
            if info['path'] == model_name or key == model_name:
                model_info = info
                break

        if not model_info:
            return jsonify({
                "success": False,
                "message": f"지원하지 않는 모델입니다: {model_name}",
                "error_type": "InvalidModel"
            }), 400

        # 4. 임시 파일 저장
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # 5. YOLODetector로 추론
            from models.yolo_detector import YOLODetector

            model_path = app.config['MODEL_DIR'] / model_info['path']
            detector = YOLODetector(
                model_path=str(model_path),
                confidence_threshold=app.config['CONFIDENCE_THRESHOLD'],
                device='cpu'  # GPU 사용 시 'cuda'로 변경
            )

            # 6. 탐지 실행
            result = detector.predict(
                image_path=temp_path,
                iou_threshold=app.config['IOU_THRESHOLD']
            )

            # 7. 결과 반환 (Pydantic 모델을 dict로 변환)
            return jsonify(result.model_dump()), 200

        finally:
            # 8. 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"❌ Error in detect_objects: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": f"서버 에러: {str(e)}",
            "error_type": "InternalError"
        }), 500


def allowed_file(filename):
    """
    파일 확장자가 허용된 형식인지 확인
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ==================== 에러 핸들러 ====================

@app.errorhandler(404)
def not_found(error):
    """404 에러 처리"""
    return jsonify({
        "success": False,
        "message": "Endpoint not found",
        "error_type": "NotFound"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 에러 처리"""
    return jsonify({
        "success": False,
        "message": "Internal server error",
        "error_type": "InternalServerError"
    }), 500


# ==================== 메인 실행 ====================

if __name__ == '__main__':
    # 개발 서버 실행
    print("=" * 50)
    print("🚀 Starting Object Detection API Server")
    print("=" * 50)
    print(f"📍 Environment: {env}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"🔧 Debug Mode: {app.config['DEBUG']}")
    print(f"🎯 Default Model: {app.config['DEFAULT_MODEL']}")
    print("=" * 50)

    app.run(
        host='0.0.0.0',  # 모든 네트워크 인터페이스에서 접근 가능
        port=5000,
        debug=app.config['DEBUG']
    )
