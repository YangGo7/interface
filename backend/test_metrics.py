"""
Advanced Dental Metrics 테스트 스크립트
실제로 계산이 되는지 확인
"""

# 간단한 Mock Detection 클래스
class MockBBox:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class MockDetection:
    def __init__(self, label, x, y, width, height, confidence=0.9):
        self.label = label
        self.bounding_box = MockBBox(x, y, width, height)
        self.confidence = confidence

# 테스트용 Detection 데이터 생성 (파노라마 X-ray 시뮬레이션)
# 상악 우측 (1x): #11, #12, #13
# 상악 좌측 (2x): #21, #22, #23
# 하악 좌측 (3x): #31, #32, #33
# 하악 우측 (4x): #41, #42, #43

test_detections = [
    # 상악 우측 (1x) - Y좌표 작음 (위쪽), X좌표는 왼쪽부터
    MockDetection("11", 100, 50, 80, 100),  # #11
    MockDetection("12", 200, 50, 80, 100),  # #12
    MockDetection("13", 300, 50, 80, 100),  # #13

    # 상악 좌측 (2x) - Y좌표 작음 (위쪽), X좌표는 오른쪽부터
    MockDetection("21", 600, 50, 80, 100),  # #21
    MockDetection("22", 500, 50, 80, 100),  # #22
    MockDetection("23", 400, 50, 80, 100),  # #23

    # 하악 좌측 (3x) - Y좌표 큼 (아래쪽), X좌표는 오른쪽부터
    MockDetection("31", 600, 200, 80, 100),  # #31
    MockDetection("32", 500, 200, 80, 100),  # #32
    MockDetection("33", 400, 200, 80, 100),  # #33

    # 하악 우측 (4x) - Y좌표 큼 (아래쪽), X좌표는 왼쪽부터
    MockDetection("41", 100, 200, 80, 100),  # #41
    MockDetection("42", 200, 200, 80, 100),  # #42
    MockDetection("43", 300, 200, 80, 100),  # #43
]

# 테스트용 GT 데이터
test_gt_data = [
    {'label': '11', 'x_center': 0.15, 'y_center': 0.15, 'width': 0.1, 'height': 0.15},
    {'label': '12', 'x_center': 0.25, 'y_center': 0.15, 'width': 0.1, 'height': 0.15},
    {'label': '21', 'x_center': 0.75, 'y_center': 0.15, 'width': 0.1, 'height': 0.15},
    {'label': '31', 'x_center': 0.75, 'y_center': 0.75, 'width': 0.1, 'height': 0.15},
    {'label': '41', 'x_center': 0.15, 'y_center': 0.75, 'width': 0.1, 'height': 0.15},
]

# 이미지 크기
img_width = 800
img_height = 300

print("=" * 80)
print("Advanced Dental Metrics Test Start")
print("=" * 80)

# 1. Sequence Accuracy 테스트
print("\n[1] Sequence Accuracy (치아 순서 정확도)")
print("-" * 80)
from utils.advanced_metric import calculate_sequence_accuracy
seq_result = calculate_sequence_accuracy(test_detections)
print(f"[OK] 순서 정확도: {seq_result['sequence_accuracy']*100:.1f}%")
print(f"   올바른 쌍: {seq_result['correct_pairs']}/{seq_result['total_pairs']}")
print(f"   사분면별 상세:")
for q, detail in seq_result['quadrant_details'].items():
    print(f"      사분면 {q}: {detail['correct']}/{detail['total']} ({detail['accuracy']*100:.1f}%)")

# 2. Inter-Tooth Distance 테스트
print("\n[2] Inter-Tooth Distance (치아 간격 검증)")
print("-" * 80)
from utils.advanced_metric import calculate_inter_tooth_distance
dist_result = calculate_inter_tooth_distance(test_detections, img_width)
print(f"[OK] 평균 치아 간격: {dist_result['mean_distance_ratio']*100:.1f}% of image width")
print(f"   정상 범위(2-10%) 내 쌍: {dist_result['validation_rate']*100:.1f}%")
print(f"   이상 간격 쌍: {dist_result['abnormal_count']}/{dist_result['total_pairs']}")
if dist_result['abnormal_pairs']:
    for pair in dist_result['abnormal_pairs'][:3]:  # 최대 3개만 출력
        print(f"      #{pair['tooth1']}-#{pair['tooth2']}: {pair['distance_ratio']*100:.1f}% ({pair['reason']})")

# 3. Anatomical Consistency 테스트
print("\n[3] Anatomical Consistency (해부학적 일관성)")
print("-" * 80)
from utils.advanced_metric import calculate_anatomical_consistency
anat_result = calculate_anatomical_consistency(test_detections)
print(f"[OK] 해부학적 일관성 점수: {anat_result['anatomical_consistency_score']*100:.1f}%")
print(f"   상악/하악 위치: {'OK 정상' if anat_result['checks']['upper_above_lower'] else 'FAIL 비정상'}")
print(f"   좌우 대칭성: {'OK 정상' if anat_result['checks']['left_right_symmetry'] else 'FAIL 비정상'} (비율: {anat_result['symmetry_ratio']:.2f})")
print(f"   치아 겹침: {'OK 정상' if anat_result['checks']['no_overlap'] else 'FAIL 비정상'} (겹침 {anat_result['overlap_count']}쌍)")

# 4. Confidence-Weighted Accuracy 테스트
print("\n4️⃣  Confidence-Weighted Accuracy (신뢰도 가중 정확도)")
print("-" * 80)
from utils.advanced_metric import calculate_confidence_weighted_accuracy
conf_result = calculate_confidence_weighted_accuracy(test_detections, test_gt_data, img_width, img_height)
print(f"✅ 신뢰도 가중 정확도: {conf_result['confidence_weighted_accuracy']*100:.1f}%")
print(f"   전체 신뢰도 합: {conf_result['total_confidence']:.2f}")

# 5. Class Balance Score 테스트
print("\n5️⃣  Class Balance Score (클래스 균형도)")
print("-" * 80)
from utils.advanced_metric import calculate_class_balance_score
balance_result = calculate_class_balance_score(test_detections, test_gt_data)
print(f"✅ 클래스 균형 점수: {balance_result['class_balance_score']:.3f}")
print(f"   클래스별 Recall 표준편차: {balance_result['class_recall_std']:.3f}")
print(f"   평균 클래스 Recall: {balance_result['mean_class_recall']*100:.1f}%")

print("\n" + "=" * 80)
print("✅ 모든 메트릭이 정상적으로 계산됩니다!")
print("=" * 80)
