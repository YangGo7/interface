# DINO+UNet 치아 탐지 API 서버

YangGo7/interface에 추가할 수 있는 DINO+UNet 기반 치아 탐지 Flask API 서버입니다.

## 📁 생성된 파일

- **predict_api.py** - 예측 로직이 함수로 리팩토링된 모듈
- **api_server.py** - Flask API 서버 메인 파일
- **test_api.py** - API 테스트 스크립트
- **requirements_api.txt** - API 서버 의존성 패키지

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements_api.txt
```

### 2. 모델 경로 설정

[api_server.py](api_server.py:19-22) 파일의 경로를 실제 환경에 맞게 수정:

```python
DINO_CONFIG = "configs\dino\DINO_4scale_cls32.py"
DINO_CHECKPOINT = r"C:\DentexSegAndDet-main\output_dino_res50_enum32\checkpoint_best_regular.pth"
UNET_CHECKPOINT = r"C:\DentexSegAndDet-main\output_unet_enum32_11-19_08-03\last_epoch.pth"
UNET_NUM_CLASSES = 33
```

### 3. API 서버 실행

```bash
python api_server.py
```

서버가 `http://localhost:5001`에서 실행됩니다.

### 4. API 테스트

```bash
python test_api.py
```

## 📡 API 엔드포인트

### 1. Health Check

```http
GET /health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. 치아 탐지 (기본)

```http
POST /api/predict/dino-unet
```

**요청 방법 1: 파일 업로드 (multipart/form-data)**

```python
import requests

with open('tooth_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'http://localhost:5001/api/predict/dino-unet',
        files=files
    )
    result = response.json()
```

**요청 방법 2: Base64 인코딩 (application/json)**

```python
import requests
import base64

with open('tooth_image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {'image': f"data:image/jpeg;base64,{image_b64}"}
response = requests.post(
    'http://localhost:5001/api/predict/dino-unet',
    json=payload
)
result = response.json()
```

**응답 예시:**
```json
{
  "success": true,
  "data": {
    "teeth": [
      {
        "fdi": "11",
        "universal": 8,
        "bbox": [245.3, 120.5, 289.7, 178.2],
        "center": [267.5, 149.35],
        "source": "UNET",
        "confidence": 1.0
      },
      {
        "fdi": "18",
        "universal": 1,
        "bbox": [50.2, 95.3, 88.9, 145.6],
        "center": [69.55, 120.45],
        "source": "DINO",
        "confidence": 0.87
      }
    ],
    "upper_arch": ["18", "17", "16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26", "27", "28"],
    "lower_arch": ["48", "47", "46", "45", "44", "43", "42", "41", "31", "32", "33", "34", "35", "36", "37", "38"],
    "total_detected": 28,
    "image_shape": [800, 1200]
  },
  "message": "Detection completed successfully"
}
```

---

### 3. 치아 탐지 + 어노테이션 이미지

```http
POST /api/predict/dino-unet/annotated
```

**요청:** 위와 동일 (파일 업로드 또는 Base64)

**응답 예시:**
```json
{
  "success": true,
  "data": { /* 위와 동일 */ },
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
  "message": "Detection and annotation completed successfully"
}
```

어노테이션된 이미지는 Base64 인코딩되어 반환되므로 프론트엔드에서 바로 표시 가능합니다.

---

## 🔗 YangGo7/interface에 통합하는 방법

### 방법 1: 독립 API 서버로 실행

1. 이 API 서버를 별도 포트(5001)에서 실행
2. YangGo7/interface의 프론트엔드에서 두 API를 모두 호출
   - 기존 YOLO: `http://localhost:5000/predict`
   - DINO+UNet: `http://localhost:5001/api/predict/dino-unet`

**프론트엔드 예시:**
```javascript
// 모델 선택
const model = document.getElementById('model-select').value;

let apiUrl;
if (model === 'yolo') {
    apiUrl = 'http://localhost:5000/predict';
} else if (model === 'dino-unet') {
    apiUrl = 'http://localhost:5001/api/predict/dino-unet';
}

// 이미지 업로드
const formData = new FormData();
formData.append('image', imageFile);

fetch(apiUrl, {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    // 결과 표시
    console.log(data);
});
```

---

### 방법 2: Interface 저장소에 라우트 추가

1. YangGo7/interface를 클론
2. `api_server.py`의 라우트를 interface의 `app.py`에 병합

```python
# YangGo7/interface의 app.py에 추가

from predict_api import TeethDetectionModel

# DINO+UNet 모델 초기화
dino_unet_model = TeethDetectionModel(
    dino_config="...",
    dino_checkpoint="...",
    unet_checkpoint="..."
)

@app.route('/api/predict/dino-unet', methods=['POST'])
def predict_dino_unet():
    # api_server.py의 로직 복사
    ...
```

---

## 🧪 테스트

### cURL로 테스트

```bash
# Health Check
curl http://localhost:5001/health

# 이미지 업로드
curl -X POST \
  -F "image=@test_image.jpg" \
  http://localhost:5001/api/predict/dino-unet
```

### Python으로 테스트

```bash
python test_api.py
```

---

## 📊 응답 데이터 구조

| 필드 | 타입 | 설명 |
|------|------|------|
| `success` | boolean | 요청 성공 여부 |
| `data.teeth` | array | 탐지된 치아 정보 배열 |
| `data.teeth[].fdi` | string | FDI 표기법 번호 (11-48) |
| `data.teeth[].universal` | int | Universal 번호 (1-32) |
| `data.teeth[].bbox` | array | 바운딩 박스 [x1, y1, x2, y2] |
| `data.teeth[].center` | array | 중심 좌표 [x, y] |
| `data.teeth[].source` | string | 탐지 모델 ("DINO" 또는 "UNET") |
| `data.teeth[].confidence` | float | 신뢰도 (0.0-1.0) |
| `data.upper_arch` | array | 상악 치아 FDI 번호 리스트 |
| `data.lower_arch` | array | 하악 치아 FDI 번호 리스트 |
| `data.total_detected` | int | 총 탐지된 치아 수 |
| `data.image_shape` | array | 이미지 크기 [height, width] |

---

## 🎨 프론트엔드 UI 예시

```html
<!DOCTYPE html>
<html>
<body>
    <h1>치아 탐지 시스템</h1>

    <select id="model-select">
        <option value="yolo">YOLO v11</option>
        <option value="dino-unet">DINO + UNet</option>
    </select>

    <input type="file" id="image-upload" accept="image/*">
    <button onclick="detectTeeth()">분석 시작</button>

    <div id="results"></div>
    <img id="annotated-image" style="display:none;">

    <script>
    async function detectTeeth() {
        const model = document.getElementById('model-select').value;
        const file = document.getElementById('image-upload').files[0];

        const apiUrl = model === 'yolo'
            ? 'http://localhost:5000/predict'
            : 'http://localhost:5001/api/predict/dino-unet/annotated';

        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // 결과 표시
            document.getElementById('results').innerHTML = `
                <p>탐지된 치아: ${result.data.total_detected}개</p>
                <p>상악: ${result.data.upper_arch.join(', ')}</p>
                <p>하악: ${result.data.lower_arch.join(', ')}</p>
            `;

            // 어노테이션 이미지 표시
            if (result.annotated_image) {
                const img = document.getElementById('annotated-image');
                img.src = result.annotated_image;
                img.style.display = 'block';
            }
        }
    }
    </script>
</body>
</html>
```

---

## ⚙️ 설정

### 포트 변경

[api_server.py](api_server.py:308) 하단:

```python
app.run(
    host='0.0.0.0',
    port=5001,  # 여기를 원하는 포트로 변경
    debug=True
)
```

### CORS 설정

다른 도메인에서 접근하려면 [api_server.py](api_server.py:17):

```python
CORS(app, origins=['http://localhost:3000', 'https://yourdomain.com'])
```

---

## 🐛 문제 해결

### 모델 로드 오류

- 체크포인트 경로가 정확한지 확인
- CUDA 사용 시 GPU 메모리 확인

### CORS 오류

- `flask-cors` 설치 확인
- 브라우저 개발자 도구에서 CORS 헤더 확인

### 포트 충돌

- 다른 프로세스가 5001 포트를 사용 중인지 확인
- 포트 변경 또는 기존 프로세스 종료

---

## 📝 라이센스

원본 프로젝트의 라이센스를 따릅니다.
