const API_URL = 'http://localhost:5000/api';
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
let currentImage = null;
let currentDetections = [];
let gtData = null;  // GT 데이터 (백엔드에서 전송)
let reportUrl = null;
let maskVisibility = {};  // {detectionId: true/false}
let allMasksVisible = true;
let gtComparisonEnabled = false;  // GT 비교 활성화 여부 (백엔드에서 처리)
let gtVisualizationEnabled = false;  // GT 시각화 On/Off
let gtLabelsVisible = true;  // GT 라벨 표시 여부

/**
 * 1. 초기화: 페이지 로드 시 모델 목록 가져오기
 */
window.onload = async () => {
    try {
        const res = await fetch(`${API_URL}/models`);
        const data = await res.json();
        
        const select = document.getElementById('modelSelect');
        select.innerHTML = ''; // 기존 옵션 초기화

        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.text = `${model.name} (${model.size})`;
            if(model.name === data.default_model) option.selected = true;
            select.appendChild(option);
        });
    } catch (e) {
        console.error(e);
        alert("서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요.");
    }
};

/**
 * 2. 이미지 미리보기 기능
 * 사용자가 파일을 선택하면 캔버스에 이미지를 미리 보여줍니다.
 */
document.getElementById('imageInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            // 캔버스 크기를 이미지 크기에 맞춤
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // 안내 문구 숨기기
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('info-panel').style.display = 'none';
        }
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

/**
 * 3. 분석 요청 및 결과 처리
 * '분석 시작' 버튼 클릭 시 호출됩니다.
 */
async function detectObjects() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files[0]) return alert("이미지를 선택해주세요!");

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('model', document.getElementById('modelSelect').value);

    // GT 파일 전송 (백엔드에서 처리)
    const gtInput = document.getElementById('gtInput');
    if (gtInput.files[0]) {
        formData.append('gt_file', gtInput.files[0]);
        console.log('GT file attached for backend processing');
    }

    // UI 상태 변경 (로딩 중 표시)
    document.getElementById('loader').style.display = 'block';
    document.getElementById('detectionBtn').disabled = true;

    try {
        const res = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            currentDetections = data.detections;
            reportUrl = data.report_url;

            // GT 데이터 저장
            gtData = data.gt_data || null;

            // GT 비교 활성화 여부 (백엔드에서 처리됨)
            gtComparisonEnabled = data.has_gt || false;

            // 마스크 가시성 초기화
            maskVisibility = {};
            data.detections.forEach(det => {
                maskVisibility[det.id] = true;
            });

            drawDetections(data.detections);
            showStats(data);
            showMissingToothAnalysis(data.analysis);
            createIndividualToggles(data.detections);

            // 컨트롤 패널 표시
            document.getElementById('controlPanel').style.display = 'block';

            // 리포트 버튼 표시
            if (reportUrl) {
                document.getElementById('reportBtn').style.display = 'inline-block';
            }

            // GT 토글 버튼 표시
            if (data.has_gt) {
                document.getElementById('gtToggleBtn').style.display = 'inline-block';
                document.getElementById('gtVisualizeBtn').style.display = 'inline-block';
                document.getElementById('gtLabelBtn').style.display = 'inline-block';
            }
        } else {
            alert('분석 실패: ' + data.message);
        }
    } catch (e) {
        console.error(e);
        alert('에러 발생: 콘솔을 확인하세요.');
    } finally {
        // UI 상태 복구
        document.getElementById('loader').style.display = 'none';
        document.getElementById('detectionBtn').disabled = false;
    }
}

/**
 * 4. 탐지 결과 그리기 (핵심 로직)
 * 마스크(Polygon)와 바운딩 박스를 캔버스에 그립니다.
 * GT 비교 시 색상 코딩 적용
 */
function drawDetections(detections) {
    // 이미지를 다시 그려서 이전 결과 지우기
    if (currentImage) {
        ctx.drawImage(currentImage, 0, 0);
    }

    // GT 시각화가 활성화된 경우 GT 마스크 먼저 그리기
    if (gtVisualizationEnabled && gtData) {
        drawGTMasks(gtData);
    }

    detections.forEach(det => {
        const { x, y, width, height } = det.bounding_box;

        // 색상 결정 (백엔드에서 처리됨)
        // GT 비교 활성화 시: gt_color 사용, 비활성화 시: original_color 사용
        let color;
        if (gtComparisonEnabled && det.gt_color) {
            color = det.gt_color;
        } else if (det.original_color) {
            color = det.original_color;
        } else {
            color = det.color || '#00FF00';
        }

        // 마스크 가시성 체크
        const isMaskVisible = maskVisibility[det.id] !== false;

        // ------------------------------------------
        // [Step A] 마스크(Segmentation Mask) 그리기
        // ------------------------------------------
        if (isMaskVisible && det.segmentation_mask && det.segmentation_mask.format === 'polygon') {
            const points = det.segmentation_mask.counts;

            if (points && points.length > 0) {
                ctx.save();

                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i][0], points[i][1]);
                }
                ctx.closePath();

                // 내부 채우기 (투명도 40%)
                ctx.fillStyle = hexToRgba(color, 0.4);
                ctx.fill();

                // 외곽선 그리기
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.restore();
            }
        }

        // ------------------------------------------
        // [Step B] 바운딩 박스(Bounding Box) 그리기
        // GT 비교 모드이거나 마스크가 보일 때만 bbox 표시
        // ------------------------------------------
        const shouldShowBbox = gtComparisonEnabled || isMaskVisible;

        if (shouldShowBbox) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
        }

        // ------------------------------------------
        // [Step C] 라벨(Label) 그리기
        // GT 비교 모드이거나 마스크가 보일 때만 라벨 표시
        // ------------------------------------------
        if (shouldShowBbox) {
            const text = `${det.label} ${Math.round(det.confidence * 100)}%`;

            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(text).width;
            const textHeight = 25;

            // 라벨 배경
            ctx.fillStyle = color;
            ctx.fillRect(x, y - textHeight, textWidth + 10, textHeight);

            // 라벨 텍스트
            ctx.fillStyle = '#fff';
            ctx.fillText(text, x + 5, y - 7);
        }
    });
}

/**
 * 5. 결과 통계 표시
 */
function showStats(data) {
    const panel = document.getElementById('info-panel');
    const { preprocessing_time_ms, inference_time_ms, total_time_ms } = data.metrics;
    
    // 탐지된 객체 수 카운트
    const counts = {};
    data.detections.forEach(d => counts[d.label] = (counts[d.label] || 0) + 1);
    
    const tagHtml = Object.entries(counts).map(([k, v]) => 
        `<span class="tag" style="background:#3498db">${k}: ${v}</span>`
    ).join('');

    panel.innerHTML = `
        <h3>📊 분석 결과</h3>
        <p><strong>총 소요 시간:</strong> ${total_time_ms}ms (추론: ${inference_time_ms}ms)</p>
        <div style="margin-top:10px">${tagHtml}</div>
    `;
    panel.style.display = 'block';
}

/**
 * Helper: Hex 색상코드를 RGBA로 변환 (투명도 적용용)
 * 예: hexToRgba('#FF0000', 0.5) -> 'rgba(255, 0, 0, 0.5)'
 */
function hexToRgba(hex, alpha) {
    // # 제거
    hex = hex.replace('#', '');

    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * 전체 마스크 토글
 */
function toggleAllMasks() {
    allMasksVisible = !allMasksVisible;

    currentDetections.forEach(det => {
        maskVisibility[det.id] = allMasksVisible;
    });

    // 체크박스 업데이트
    document.querySelectorAll('.mask-checkbox').forEach(checkbox => {
        checkbox.checked = allMasksVisible;
    });

    // 버튼 텍스트 변경
    document.getElementById('maskToggleBtn').textContent =
        allMasksVisible ? 'Hide All Masks' : 'Show All Masks';

    drawDetections(currentDetections);
}

/**
 * 개별 마스크 토글
 */
function toggleIndividualMask(detectionId) {
    maskVisibility[detectionId] = !maskVisibility[detectionId];
    drawDetections(currentDetections);
}

/**
 * 개별 토글 체크박스 생성
 */
function createIndividualToggles(detections) {
    const container = document.getElementById('individualToggles');
    container.innerHTML = '<h4 style="margin-bottom:10px;">Individual Mask Control:</h4>';

    detections.forEach(det => {
        const div = document.createElement('div');
        div.style.marginBottom = '5px';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.className = 'mask-checkbox';
        checkbox.onchange = () => toggleIndividualMask(det.id);

        const label = document.createElement('label');
        label.style.marginLeft = '5px';
        label.textContent = `${det.label} (${Math.round(det.confidence * 100)}%)`;

        div.appendChild(checkbox);
        div.appendChild(label);
        container.appendChild(div);
    });
}

/**
 * 리포트 열기
 */
function openReport() {
    if (reportUrl) {
        window.open(reportUrl, '_blank');
    } else {
        alert('리포트가 생성되지 않았습니다.');
    }
}

/**
 * 결손치 분석 결과 표시
 */
function showMissingToothAnalysis(analysis) {
    if (!analysis) return;

    const panel = document.getElementById('analysisPanel');
    const infoDiv = document.getElementById('missingToothInfo');

    const detectedList = analysis.detected.join(', ') || 'None';
    const missingList = analysis.missing.join(', ') || 'None';

    infoDiv.innerHTML = `
        <p><strong>Status:</strong> ${analysis.status}</p>
        <p><strong>Detected Teeth:</strong> ${detectedList}</p>
        <p><strong>Missing Teeth:</strong> ${missingList} (${analysis.missing_count} total)</p>
    `;

    panel.style.display = 'block';
}

/**
 * GT 마스크 그리기
 * YOLO 포맷의 normalized 좌표를 pixel 좌표로 변환하여 마스크로 표시합니다.
 * - Segmentation 포맷: polygon으로 그리기
 * - BBox 포맷: 채워진 사각형으로 그리기
 */
function drawGTMasks(gtObjects) {
    if (!gtObjects || gtObjects.length === 0) return;

    const imgWidth = canvas.width;
    const imgHeight = canvas.height;

    // GT 마스크 색상: 파란색 (Detection과 구분)
    const gtColor = '#0000FF';
    const gtAlpha = 0.3;  // 투명도

    gtObjects.forEach(gt => {
        ctx.save();

        // Segmentation 포맷 (polygon)
        if (gt.type === 'segmentation' && gt.polygon) {
            // Normalized → Pixel 좌표 변환
            const pixelPolygon = gt.polygon.map(([x, y]) => [
                x * imgWidth,
                y * imgHeight
            ]);

            if (pixelPolygon.length >= 3) {
                // Polygon 그리기
                ctx.beginPath();
                ctx.moveTo(pixelPolygon[0][0], pixelPolygon[0][1]);
                for (let i = 1; i < pixelPolygon.length; i++) {
                    ctx.lineTo(pixelPolygon[i][0], pixelPolygon[i][1]);
                }
                ctx.closePath();

                // 내부 채우기 (투명도 적용)
                ctx.fillStyle = hexToRgba(gtColor, gtAlpha);
                ctx.fill();

                // 외곽선 그리기
                ctx.strokeStyle = gtColor;
                ctx.lineWidth = 2;
                ctx.stroke();

                // 라벨 그리기 (gtLabelsVisible이 true일 때만)
                if (gtLabelsVisible) {
                    const labelX = pixelPolygon[0][0];
                    const labelY = pixelPolygon[0][1];

                    const text = `GT: ${gt.label}`;
                    ctx.font = '14px Arial';
                    const textWidth = ctx.measureText(text).width;
                    const textHeight = 20;

                    ctx.fillStyle = gtColor;
                    ctx.fillRect(labelX, labelY - textHeight, textWidth + 8, textHeight);

                    ctx.fillStyle = '#fff';
                    ctx.fillText(text, labelX + 4, labelY - 5);
                }
            }
        }
        // BBox 포맷 (채워진 사각형)
        else if (gt.type === 'bbox' || (!gt.type && gt.x_center !== undefined)) {
            // Normalized → Pixel 좌표 변환
            const x = (gt.x_center - gt.width / 2) * imgWidth;
            const y = (gt.y_center - gt.height / 2) * imgHeight;
            const w = gt.width * imgWidth;
            const h = gt.height * imgHeight;

            // 채워진 사각형 그리기
            ctx.fillStyle = hexToRgba(gtColor, gtAlpha);
            ctx.fillRect(x, y, w, h);

            // 외곽선 그리기
            ctx.strokeStyle = gtColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // 라벨 그리기 (gtLabelsVisible이 true일 때만)
            if (gtLabelsVisible) {
                const text = `GT: ${gt.label}`;
                ctx.font = '14px Arial';
                const textWidth = ctx.measureText(text).width;
                const textHeight = 20;

                ctx.fillStyle = gtColor;
                ctx.fillRect(x, y - textHeight, textWidth + 8, textHeight);

                ctx.fillStyle = '#fff';
                ctx.fillText(text, x + 4, y - 5);
            }
        }

        ctx.restore();
    });
}

/**
 * GT 시각화 토글
 */
function toggleGTVisualization() {
    if (!gtData) {
        alert('GT 파일이 로드되지 않았습니다.');
        return;
    }

    gtVisualizationEnabled = !gtVisualizationEnabled;

    // 버튼 텍스트 변경
    document.getElementById('gtVisualizeBtn').textContent =
        gtVisualizationEnabled ? 'Hide GT Masks' : 'Show GT Masks';

    drawDetections(currentDetections);
}

/**
 * GT 라벨 토글
 */
function toggleGTLabels() {
    if (!gtData) {
        alert('GT 파일이 로드되지 않았습니다.');
        return;
    }

    gtLabelsVisible = !gtLabelsVisible;

    // 버튼 텍스트 변경
    document.getElementById('gtLabelBtn').textContent =
        gtLabelsVisible ? 'Hide GT Labels' : 'Show GT Labels';

    drawDetections(currentDetections);
}

/**
 * GT 비교 토글
 * 백엔드에서 이미 GT 비교를 수행했으므로,
 * 프론트엔드에서는 gt_color와 original_color 간 전환만 수행
 */
function toggleGTComparison() {
    // currentDetections에 gt_color가 있는지 확인
    const hasGTColors = currentDetections.some(det => det.gt_color !== undefined);

    if (!hasGTColors) {
        alert('GT 파일이 로드되지 않았습니다.');
        return;
    }

    gtComparisonEnabled = !gtComparisonEnabled;

    // 버튼 텍스트 변경
    document.getElementById('gtToggleBtn').textContent =
        gtComparisonEnabled ? 'Disable GT Comparison' : 'Enable GT Comparison';

    drawDetections(currentDetections);
}