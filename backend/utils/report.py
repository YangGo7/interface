import cv2
import numpy as np
import base64
import os
from typing import List, Dict, Any, Optional
from collections import Counter
import datetime

# Advanced Dental Metrics 모듈 import
from utils.advanced_metric import (
    calculate_sequence_accuracy,
    calculate_inter_tooth_distance,
    calculate_anatomical_consistency,
    calculate_confidence_weighted_accuracy,
    calculate_class_balance_score
)

class ReportGenerator:
    """
    HTML 리포트 생성 클래스
    - 원본/예측 이미지 시각화 포함
    - 전체 및 클래스별 성능 지표 테이블 포함
    """
    
    def __init__(self, detections: List[Any], model_name: str, image_path: str, gt_data: Optional[List[Dict[str, Any]]] = None):
        self.detections = detections
        self.model_name = model_name
        self.image_path = image_path
        self.gt_data = gt_data
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 이미지 로드 (시각화용)
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.img_height, self.img_width = self.image.shape[:2]

    def _encode_image_to_base64(self, img) -> str:
        """OpenCV 이미지를 Base64 문자열로 변환 (HTML 임베딩용)"""
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def calculate_metrics_with_gt(self) -> Dict[str, Any]:
        """
        GT 데이터가 있을 때 mAP, IoU, Precision, Recall 등을 계산합니다.

        Returns:
            Dict with metrics: mAP, mean_iou, precision, recall, f1_score, tp, fp, fn
        """
        if not self.gt_data or len(self.gt_data) == 0:
            return None

        from utils.gt_comparison import find_best_gt_match

        # 매칭된 GT 추적 (중복 매칭 방지)
        matched_gt_indices = set()

        tp = 0  # True Positives
        fp = 0  # False Positives
        iou_list = []

        # IoU 임계값 (일반적으로 0.5 사용)
        iou_threshold = 0.5

        # 각 detection에 대해 GT 매칭
        for detection in self.detections:
            match_result = find_best_gt_match(detection, self.gt_data, self.img_width, self.img_height)

            if match_result and match_result['iou'] >= iou_threshold:
                # IoU가 임계값 이상이면서 라벨이 일치하는 경우만 TP
                if match_result['label_match']:
                    tp += 1
                    iou_list.append(match_result['iou'])

                    # 매칭된 GT 인덱스 추적 (중복 방지)
                    # GT 리스트에서 해당 GT의 인덱스 찾기
                    for idx, gt in enumerate(self.gt_data):
                        if gt['label'] == match_result['gt_label']:
                            matched_gt_indices.add(idx)
                            break
                else:
                    # IoU는 높지만 라벨이 틀림 -> FP
                    fp += 1
            else:
                # IoU가 낮거나 매칭 없음 -> FP
                fp += 1

        # False Negatives: GT에는 있지만 detection 안됨
        fn = len(self.gt_data) - len(matched_gt_indices)

        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0

        # mAP 계산 (simplified version - AP@0.5)
        # 실제 mAP는 여러 IoU threshold에서 계산하지만, 여기서는 0.5에서만 계산
        mAP = precision  # AP@0.5 = Precision at IoU threshold 0.5

        # ===== 고급 치식 탐지 지표 계산 =====
        # 기본 메트릭 외에 치아 순서, 간격, 해부학적 일관성 등을 추가로 계산

        # 1. 순서 정확도: 사분면 내 치아가 올바른 순서로 배열되었는지
        sequence_metrics = calculate_sequence_accuracy(self.detections)

        # 2. 치아 간격 검증: 인접 치아 간 거리가 정상 범위인지
        distance_metrics = calculate_inter_tooth_distance(self.detections, self.img_width)

        # 3. 해부학적 일관성: 상하악 위치, 좌우 대칭성, 겹침 검사
        anatomy_metrics = calculate_anatomical_consistency(self.detections)

        # 4. 신뢰도 가중 정확도: confidence로 가중 평균한 정확도
        confidence_metrics = calculate_confidence_weighted_accuracy(
            self.detections, self.gt_data, self.img_width, self.img_height
        )

        # 5. 클래스 균형도: 모든 치아를 골고루 잘 탐지하는지
        balance_metrics = calculate_class_balance_score(self.detections, self.gt_data)

        # 모든 메트릭을 하나의 딕셔너리로 통합하여 반환
        return {
            # 기본 Detection 메트릭
            'mAP': mAP,
            'mean_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_detections': len(self.detections),
            'total_gt': len(self.gt_data),

            # 고급 치식 탐지 메트릭
            'sequence_accuracy': sequence_metrics['sequence_accuracy'],
            'sequence_correct_pairs': sequence_metrics['correct_pairs'],
            'sequence_total_pairs': sequence_metrics['total_pairs'],
            'sequence_quadrant_details': sequence_metrics['quadrant_details'],

            'inter_tooth_mean_distance': distance_metrics['mean_distance_ratio'],
            'inter_tooth_abnormal_count': distance_metrics['abnormal_count'],
            'inter_tooth_validation_rate': distance_metrics['validation_rate'],
            'inter_tooth_abnormal_pairs': distance_metrics['abnormal_pairs'],

            'anatomical_consistency': anatomy_metrics['anatomical_consistency_score'],
            'anatomical_checks': anatomy_metrics['checks'],
            'anatomical_symmetry_ratio': anatomy_metrics['symmetry_ratio'],
            'anatomical_overlap_count': anatomy_metrics['overlap_count'],

            'confidence_weighted_accuracy': confidence_metrics['confidence_weighted_accuracy'],
            'total_confidence': confidence_metrics['total_confidence'],

            'class_balance_score': balance_metrics['class_balance_score'],
            'class_recall_std': balance_metrics['class_recall_std'],
            'mean_class_recall': balance_metrics['mean_class_recall']
        }

    def _draw_predictions(self) -> np.ndarray:
        """원본 이미지 위에 마스크와 박스를 그립니다."""
        vis_img = self.image.copy()
        
        # 마스크 오버레이를 위한 빈 캔버스
        mask_overlay = np.zeros_like(vis_img)
        
        for det in self.detections:
            # 1. 색상 결정 (Hex -> BGR)
            hex_color = det.color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color = (b, g, r) # OpenCV uses BGR

            # 2. 마스크 그리기
            if det.segmentation_mask and det.segmentation_mask.counts:
                points = np.array(det.segmentation_mask.counts, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [points], color)

            # 3. 바운딩 박스 그리기
            x, y = int(det.bounding_box.x), int(det.bounding_box.y)
            w, h = int(det.bounding_box.width), int(det.bounding_box.height)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)

            # 4. 라벨 텍스트
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(vis_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 원본과 마스크 합성 (투명도 적용)
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.4, 0)
        return vis_img

    def save_html_report(self, save_path: str, metrics: Dict[str, float] = None, analysis_result: Dict = None) -> str:
        """
        HTML 리포트를 생성하고 파일로 저장합니다.
        Args:
            save_path: 저장할 파일 경로 (예: ./reports/result.html)
            metrics: 외부에서 계산된 IoU, mAP, Accuracy 등 (없으면 N/A 처리)
            analysis_result: 결손치 분석 결과
        """
        # 1. 이미지 준비 (Base64 인코딩)
        img_original_b64 = self._encode_image_to_base64(self.image)
        img_prediction = self._draw_predictions()
        img_prediction_b64 = self._encode_image_to_base64(img_prediction)

        # 2. 통계 계산
        total_obj = len(self.detections)
        labels = [det.label for det in self.detections]
        counts = dict(Counter(labels))
        
        # 평균 신뢰도 계산
        avg_conf = sum([d.confidence for d in self.detections]) / total_obj if total_obj > 0 else 0.0

        # GT 데이터가 있으면 자동으로 메트릭 계산
        if metrics is None and self.gt_data:
            calculated_metrics = self.calculate_metrics_with_gt()
            if calculated_metrics:
                # 기본 Detection 메트릭 포맷팅
                metrics = {
                    "mAP": f"{calculated_metrics['mAP']*100:.2f}%",
                    "IoU": f"{calculated_metrics['mean_iou']:.3f}",
                    "Precision": f"{calculated_metrics['precision']*100:.2f}%",
                    "Recall": f"{calculated_metrics['recall']*100:.2f}%",
                    "F1-Score": f"{calculated_metrics['f1_score']:.3f}",
                    "TP": calculated_metrics['tp'],
                    "FP": calculated_metrics['fp'],
                    "FN": calculated_metrics['fn'],

                    # 고급 치식 탐지 메트릭 추가
                    "Sequence Accuracy": f"{calculated_metrics['sequence_accuracy']*100:.1f}%",
                    "Sequence Pairs": f"{calculated_metrics['sequence_correct_pairs']}/{calculated_metrics['sequence_total_pairs']}",

                    "Inter-Tooth Distance": f"{calculated_metrics['inter_tooth_mean_distance']*100:.1f}% of img width",
                    "Distance Validation": f"{calculated_metrics['inter_tooth_validation_rate']*100:.1f}%",
                    "Abnormal Gaps": calculated_metrics['inter_tooth_abnormal_count'],

                    "Anatomical Consistency": f"{calculated_metrics['anatomical_consistency']*100:.1f}%",
                    "Symmetry Ratio": f"{calculated_metrics['anatomical_symmetry_ratio']:.2f}",
                    "Bbox Overlaps": calculated_metrics['anatomical_overlap_count'],

                    "Confidence-Weighted Acc": f"{calculated_metrics['confidence_weighted_accuracy']*100:.1f}%",

                    "Class Balance Score": f"{calculated_metrics['class_balance_score']:.3f}",
                    "Class Recall Std": f"{calculated_metrics['class_recall_std']:.3f}",

                    # 원본 메트릭 데이터도 보존 (상세 분석용)
                    "_raw": calculated_metrics
                }
            else:
                metrics = {
                    "mAP": "N/A (No GT)",
                    "IoU": "N/A (No GT)",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "Sequence Accuracy": "N/A",
                    "Inter-Tooth Distance": "N/A",
                    "Anatomical Consistency": "N/A",
                    "Confidence-Weighted Acc": "N/A",
                    "Class Balance Score": "N/A"
                }
        elif metrics is None:
            # GT 없으면 기본값
            metrics = {
                "mAP": "N/A (No GT)",
                "IoU": "N/A (No GT)",
                "Precision": "N/A",
                "Recall": "N/A",
                "Sequence Accuracy": "N/A",
                "Inter-Tooth Distance": "N/A",
                "Anatomical Consistency": "N/A",
                "Confidence-Weighted Acc": "N/A",
                "Class Balance Score": "N/A"
            }

        # 3. HTML 생성
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Dental Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f4f4f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 10px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .header-info {{ display: flex; justify-content: space-between; margin-bottom: 20px; color: #555; }}
                
                /* 이미지 영역 */
                .image-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
                .img-box {{ text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                
                /* 테이블 스타일 */
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                
                .metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .badge {{ display: inline-block; padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.8em; }}
                .bg-green {{ background: #2ecc71; }}
                .bg-red {{ background: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🦷 AI Dental Diagnosis Report</h1>
                <div class="header-info">
                    <span><strong>Model:</strong> {self.model_name}</span>
                    <span><strong>Date:</strong> {self.timestamp}</span>
                </div>

                <h2>1. Visual Analysis</h2>
                <div class="image-grid">
                    <div class="img-box">
                        <h3>Original Image</h3>
                        <img src="data:image/jpeg;base64,{img_original_b64}" alt="Original">
                    </div>
                    <div class="img-box">
                        <h3>Prediction (Masks & BBox)</h3>
                        <img src="data:image/jpeg;base64,{img_prediction_b64}" alt="Prediction">
                    </div>
                </div>

                <h2>2. Advanced Dental Metrics</h2>
                <div class="metric-box">
                    <p><strong>Note:</strong> 치식 탐지의 정확성을 평가하기 위한 고급 지표입니다. 순서, 간격, 해부학적 일관성을 검증합니다.</p>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>Sequence Accuracy (순서 정확도)</td>
                            <td>{metrics.get('Sequence Accuracy', 'N/A')}</td>
                            <td>사분면 내 치아가 올바른 X좌표 순서로 배열된 비율</td>
                        </tr>
                        <tr>
                            <td>Sequence Pairs (순서 검증 쌍)</td>
                            <td>{metrics.get('Sequence Pairs', 'N/A')}</td>
                            <td>올바른 순서 쌍 / 전체 인접 쌍</td>
                        </tr>
                        <tr>
                            <td>Inter-Tooth Distance (평균 치아 간격)</td>
                            <td>{metrics.get('Inter-Tooth Distance', 'N/A')}</td>
                            <td>인접 치아 간 평균 거리 (이미지 너비 대비)</td>
                        </tr>
                        <tr>
                            <td>Distance Validation (간격 검증율)</td>
                            <td>{metrics.get('Distance Validation', 'N/A')}</td>
                            <td>정상 범위(2-10%) 내 치아 쌍 비율</td>
                        </tr>
                        <tr>
                            <td>Abnormal Gaps (이상 간격 쌍)</td>
                            <td>{metrics.get('Abnormal Gaps', 'N/A')}</td>
                            <td>너무 가깝거나 먼 인접 치아 쌍 개수</td>
                        </tr>
                        <tr>
                            <td>Anatomical Consistency (해부학적 일관성)</td>
                            <td>{metrics.get('Anatomical Consistency', 'N/A')}</td>
                            <td>상하악 위치, 좌우 대칭성, 겹침 검사 종합 점수</td>
                        </tr>
                        <tr>
                            <td>Symmetry Ratio (좌우 대칭 비율)</td>
                            <td>{metrics.get('Symmetry Ratio', 'N/A')}</td>
                            <td>좌측 치아 수 / 우측 치아 수 (1.0이 완전 대칭)</td>
                        </tr>
                        <tr>
                            <td>Bbox Overlaps (박스 겹침 개수)</td>
                            <td>{metrics.get('Bbox Overlaps', 'N/A')}</td>
                            <td>IoU > 0.3인 치아 쌍 개수 (0이 정상)</td>
                        </tr>
                        <tr>
                            <td>Confidence-Weighted Accuracy (신뢰도 가중 정확도)</td>
                            <td>{metrics.get('Confidence-Weighted Acc', 'N/A')}</td>
                            <td>높은 confidence 예측을 더 중요하게 평가</td>
                        </tr>
                        <tr>
                            <td>Class Balance Score (클래스 균형도)</td>
                            <td>{metrics.get('Class Balance Score', 'N/A')}</td>
                            <td>모든 치아를 골고루 탐지하는 정도 (1.0이 완전 균형)</td>
                        </tr>
                        <tr>
                            <td>Class Recall Std (클래스별 Recall 표준편차)</td>
                            <td>{metrics.get('Class Recall Std', 'N/A')}</td>
                            <td>클래스별 성능 편차 (0에 가까울수록 균형적)</td>
                        </tr>
                    </table>
                </div>

                <h2>3. Overall Detection Metrics</h2>
                <div class="metric-box">
                    <p><strong>Note:</strong> Real-time metrics (IoU, mAP) require Ground Truth labels. Currently showing inference results.</p>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>Total Objects Detected</td>
                            <td>{total_obj}</td>
                            <td>Number of teeth/objects found</td>
                        </tr>
                        <tr>
                            <td>Average Confidence</td>
                            <td>{avg_conf*100:.2f}%</td>
                            <td>Mean prediction probability</td>
                        </tr>
                        <tr>
                            <td>mAP@0.5 (Mean Average Precision)</td>
                            <td>{metrics.get('mAP', 'N/A')}</td>
                            <td>Overall detection accuracy at IoU=0.5</td>
                        </tr>
                        <tr>
                            <td>Mean IoU</td>
                            <td>{metrics.get('IoU', 'N/A')}</td>
                            <td>Average Intersection over Union</td>
                        </tr>
                        <tr>
                            <td>Precision</td>
                            <td>{metrics.get('Precision', 'N/A')}</td>
                            <td>TP / (TP + FP)</td>
                        </tr>
                        <tr>
                            <td>Recall</td>
                            <td>{metrics.get('Recall', 'N/A')}</td>
                            <td>TP / (TP + FN)</td>
                        </tr>
                        <tr>
                            <td>F1-Score</td>
                            <td>{metrics.get('F1-Score', 'N/A')}</td>
                            <td>2 * (Precision * Recall) / (Precision + Recall)</td>
                        </tr>
                        <tr>
                            <td>True Positives (TP)</td>
                            <td>{metrics.get('TP', 'N/A')}</td>
                            <td>Correct detections</td>
                        </tr>
                        <tr>
                            <td>False Positives (FP)</td>
                            <td>{metrics.get('FP', 'N/A')}</td>
                            <td>Incorrect detections</td>
                        </tr>
                        <tr>
                            <td>False Negatives (FN)</td>
                            <td>{metrics.get('FN', 'N/A')}</td>
                            <td>Missed GT objects</td>
                        </tr>
                    </table>
                </div>

                <h2>4. Class-wise Analysis</h2>
                <table>
                    <tr>
                        <th>Class Label</th>
                        <th>Count</th>
                        <th>Avg Confidence</th>
                        <th>Status</th>
                    </tr>
        """
        
        # 클래스별 통계 행 추가
        for label, count in counts.items():
            # 해당 클래스의 평균 conf 계산
            class_confs = [d.confidence for d in self.detections if d.label == label]
            class_avg = sum(class_confs) / len(class_confs) if class_confs else 0
            
            html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{count}</td>
                        <td>{class_avg*100:.1f}%</td>
                        <td><span class="badge bg-green">Detected</span></td>
                    </tr>
            """
            
        # 결손치 정보 추가 (Analysis Result가 있는 경우)
        if analysis_result and 'missing' in analysis_result:
            for missing_id in analysis_result['missing']:
                html_content += f"""
                    <tr>
                        <td>{missing_id}</td>
                        <td>0</td>
                        <td>-</td>
                        <td><span class="badge bg-red">Missing</span></td>
                    </tr>
                """

        html_content += """
                </table>
                
                <div style="text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em;">
                    Generated by Object Detection API
                </div>
            </div>
        </body>
        </html>
        """

        # 4. 파일 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return save_path