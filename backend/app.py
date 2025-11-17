"""
Flask Application - Object Detection API
YOLO 기반 오브젝트 디텍션 API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from config import config

# Flask 앱 생성
app = Flask(__name__)

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

@app.route('/', methods=['GET'])
def index():
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
