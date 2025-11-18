"""
Flask Application - Object Detection API
YOLO 기반 오브젝트 디텍션 API 서버 (통합 후처리 및 리포트 생성)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from config import config

from models.yolo_detector import YOLODetector
from utils.post_processing import ObjectCropper, MissingToothFinder
from utils.report import ReportGenerator

# ---------------------------------------------------------
# 1. 앱 초기화 및 설정
# ---------------------------------------------------------
app = Flask(__name__,
            static_folder=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../frontend'),
            static_url_path='')

# 환경 설정 로드
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# CORS 설정
CORS(app, resources={
    r"/api/*": {
        "origins": app.config['CORS_ORIGINS'],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ---------------------------------------------------------
# 2. 기본 라우트
# ---------------------------------------------------------

@app.route("/")
def index():
    """프론트엔드 메인 페이지 반환"""
    return send_from_directory(app.static_folder, 'index.html')

# 임시 파일 서빙 (크롭 이미지, 리포트 등)
@app.route('/temp/<path:filename>')
def serve_temp_files(filename):
    temp_dir = os.path.join(str(app.config['BASE_DIR']), 'temp')
    return send_from_directory(temp_dir, filename)

@app.route('/api/health', methods=['GET'])
def health():
    """API 상태 확인"""
    return jsonify({"status": "ok", "message": "API is working"})

@app.route('/api/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록 반환"""
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

# ---------------------------------------------------------
# 3. 객체 탐지 API (핵심 로직)
# ---------------------------------------------------------

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """
    이미지를 받아 객체를 탐지하고,
    결손치 분석, 크롭, HTML 리포트를 생성하여 반환합니다.
    """
    temp_path = None

    try:
        # A. 요청 검증
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "이미지 파일이 없습니다.", "error_type": "NoImageFile"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "message": "파일이 선택되지 않았습니다.", "error_type": "EmptyFilename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": f"지원하지 않는 파일 형식입니다.", "error_type": "InvalidFileFormat"}), 400

        # B. 모델 선택
        model_name = request.form.get('model', app.config['DEFAULT_MODEL'])
        model_info = app.config['SUPPORTED_MODELS'].get(model_name)

        if not model_info:
            # 키로 못 찾으면 path로 한 번 더 검색
            for key, info in app.config['SUPPORTED_MODELS'].items():
                if info['path'] == model_name:
                    model_info = info
                    break

        if not model_info:
            return jsonify({"success": False, "message": f"지원하지 않는 모델입니다: {model_name}", "error_type": "InvalidModel"}), 400

        # C. 케이스별 폴더 생성 및 임시 파일 저장
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_folder_name = f"case_{timestamp}"

        base_temp_dir = os.path.join(str(app.config['BASE_DIR']), 'temp')
        case_dir = os.path.join(base_temp_dir, case_folder_name)
        os.makedirs(case_dir, exist_ok=True)

        # 이미지 파일 저장
        original_ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(case_dir, f"original{original_ext}")
        file.save(temp_path)

        # GT 파일 처리 (있으면)
        gt_file = request.files.get('gt_file')
        gt_data = None
        if gt_file and gt_file.filename != '':
            gt_path = os.path.join(case_dir, 'gt.txt')
            gt_file.save(gt_path)
            from utils.gt_comparison import parse_gt_file
            gt_data = parse_gt_file(gt_path)
            print(f"📋 GT file loaded: {len(gt_data)} objects")

        # D. 추론 실행 (Inference)
        from pathlib import Path
        model_dir = Path(app.config['BASE_DIR']) / 'weights'
        model_path = model_dir / model_info['path']

        print(f"📂 Model directory: {model_dir}")
        print(f"📄 Model path: {model_path}")

        # Detector 인스턴스 생성
        detector = YOLODetector(
            model_path=str(model_path),
            confidence_threshold=float(request.form.get('conf', app.config['CONFIDENCE_THRESHOLD'])),
            device='cpu' # 필요시 'cuda'로 변경
        )

        # 예측 수행 (파라미터: NMS, 리사이즈, 마스크 품질 등)
        result = detector.predict(
            image_path=temp_path,
            iou_threshold=float(request.form.get('iou', app.config['IOU_THRESHOLD'])),
            imgsz=int(request.form.get('imgsz', 1280)),
            retina_masks=True
        )

        # GT 비교 수행 (GT 데이터가 있을 경우)
        if gt_data and len(gt_data) > 0:
            from utils.gt_comparison import find_best_gt_match, get_color_by_match_quality

            # 이미지 크기 가져오기
            img_width = result.image_info.width
            img_height = result.image_info.height

            print(f"🔍 Performing GT comparison for {len(result.detections)} detections...")

            for detection in result.detections:
                # 원본 색상 저장
                detection.original_color = detection.color

                # GT와 매칭
                match_result = find_best_gt_match(detection, gt_data, img_width, img_height)

                if match_result:
                    # GT 정보 저장
                    detection.gt_iou = match_result['iou']
                    detection.gt_label_match = match_result['label_match']

                    # GT 비교 색상 결정
                    detection.gt_color = get_color_by_match_quality(match_result)

                    # 기본 색상을 GT 색상으로 변경
                    detection.color = detection.gt_color

                    print(f"  ✓ Detection {detection.id} (pred: {detection.label}, gt: {match_result['gt_label']}): IoU={match_result['iou']:.2f}, Match={match_result['label_match']} → {detection.gt_color}")
                else:
                    # GT 매칭 없음
                    detection.gt_color = detection.color
                    print(f"  - Detection {detection.id} ({detection.label}): No GT match")

            # GT 비교 메트릭 요약 출력
            print("\n📊 GT Comparison Metrics Summary:")
            tp = sum(1 for det in result.detections if hasattr(det, 'gt_label_match') and det.gt_label_match and hasattr(det, 'gt_iou') and det.gt_iou >= 0.5)
            fp = len(result.detections) - tp
            fn = len(gt_data) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            mean_iou = sum(det.gt_iou for det in result.detections if hasattr(det, 'gt_iou')) / len(result.detections) if result.detections else 0.0

            print(f"  - Precision: {precision*100:.2f}% (TP={tp}, FP={fp})")
            print(f"  - Recall: {recall*100:.2f}% (FN={fn})")
            print(f"  - F1-Score: {f1:.3f}")
            print(f"  - Mean IoU: {mean_iou:.3f}")
            print(f"  - mAP@0.5: {precision*100:.2f}%\n")

        # 기본 응답 데이터 구성
        response_data = result.model_dump()
        response_data['has_gt'] = gt_data is not None and len(gt_data) > 0

        # GT 데이터를 프론트엔드로 전송 (시각화용)
        if gt_data:
            response_data['gt_data'] = gt_data

        # ------------------------------------------------------
        # E. 후처리 파이프라인 (Post-Processing Pipeline)
        # ------------------------------------------------------

        # 1. [결손치 분석] Missing Tooth Analysis
        analysis_result = None
        if len(result.detections) > 0:
            # MissingToothFinder는 함수이므로 직접 호출하여 결과를 받음
            analysis_result = MissingToothFinder(result.detections)
            response_data['analysis'] = analysis_result

        # 2. [이미지 크롭] Object Cropping
        # 크롭 이미지를 케이스 폴더 내 crops 하위 폴더에 저장
        crop_dir = os.path.join(case_dir, 'crops')
        os.makedirs(crop_dir, exist_ok=True)

        if len(result.detections) > 0:
            cropper = ObjectCropper(temp_path, result.detections)
            # 치근 확인을 위해 'box' 모드 권장 (배경 포함)
            crop_mode = request.form.get('crop_mode', 'box')
            cropped_files = cropper.run(crop_dir, mode=crop_mode)

            # 크롭 파일 경로를 케이스 폴더 경로로 업데이트
            for crop in cropped_files:
                crop['path'] = f"/temp/{case_folder_name}/crops/{crop['filename']}"

            response_data['crops'] = cropped_files

        # 3. [HTML 리포트 생성] Report Generation
        # 리포트를 케이스 폴더에 직접 저장
        report_filename = "report.html"
        report_save_path = os.path.join(case_dir, report_filename)

        # Reporter 인스턴스 생성 (이미지 경로 및 GT 데이터 전달)
        reporter = ReportGenerator(
            detections=result.detections,
            model_name=model_name,
            image_path=temp_path,
            gt_data=gt_data  # GT 파일이 있으면 자동으로 메트릭 계산
        )

        # HTML 저장
        reporter.save_html_report(
            save_path=report_save_path,
            analysis_result=analysis_result,
            # 추론 단계에서는 정답 라벨이 없으므로 Metrics는 N/A로 표시됩니다.
            # 만약 외부에서 계산된 값이 있다면 여기에 딕셔너리로 전달하세요.
            metrics=None
        )

        # 응답에 리포트 URL 추가
        response_data['report_url'] = f"/temp/{case_folder_name}/{report_filename}"

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Error in detect_objects: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"서버 처리 중 오류가 발생했습니다: {str(e)}",
            "error_type": "InternalError"
        }), 500

    finally:
        # F. 케이스 폴더 유지 (리포트, 크롭 이미지 접근을 위해)
        # 케이스 폴더는 삭제하지 않음 - 사용자가 리포트와 크롭 이미지를 확인할 수 있도록
        # 필요시 별도의 정리 스크립트로 오래된 케이스 폴더를 삭제할 수 있음
        pass

# ---------------------------------------------------------
# 4. 유틸리티 및 에러 핸들러
# ---------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "message": "Endpoint not found", "error_type": "NotFound"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "message": "Internal server error", "error_type": "InternalServerError"}), 500

# ---------------------------------------------------------
# 5. 서버 실행
# ---------------------------------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting Object Detection API Server")
    print("=" * 50)
    print(f"📍 Environment: {env}")
    print(f"🌐 Server: http://localhost:5000")
    print(f"🔧 Debug Mode: {app.config['DEBUG']}")
    print(f"🎯 Default Model: {app.config['DEFAULT_MODEL']}")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )