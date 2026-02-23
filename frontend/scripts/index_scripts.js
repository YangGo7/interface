const API_URL = `${window.location.origin}/api`;
const exportBtn = document.getElementById('exportBtn');
const uploadLabel = document.getElementById('uploadLabel');

let panoStatusTimer = null;
let currentOverlayUrl = null;

window.onload = async () => {
  await loadModels();
  bindInput();
};

async function loadModels() {
  try {
    const res = await fetch(`${API_URL}/models`);
    const data = await res.json();
    const select = document.getElementById('modelSelect');
    select.innerHTML = '';
    data.models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.text = `${m.name} (${m.size})`;
      if (m.name === data.default_model) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error(e);
  }
}

function bindInput() {
  const fileInput = document.getElementById('imageInput');
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    uploadLabel.textContent = file ? file.name : '이미지/DICOM input';
    currentOverlayUrl = null;
    if (exportBtn) exportBtn.disabled = true;
  });
}

async function detectObjects() {
  const fileInput = document.getElementById('imageInput');
  const modelSelect = document.getElementById('modelSelect');
  if (!fileInput.files[0]) return alert('이미지를 선택해주세요!');

  document.getElementById('detectionBtn').disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    if (modelSelect && modelSelect.value) formData.append('model', modelSelect.value);

    const res = await fetch(`${API_URL}/detect`, { method: 'POST', body: formData });
    const data = await res.json();
    if (!data.success) {
      alert('제출 실패: ' + (data.message || 'unknown'));
      return;
    }
    const statusUrlRaw = data.status_url || data.statusUrl || (data.job_id ? `/api/detect/status/${data.job_id}` : null);
    if (!statusUrlRaw) {
      alert('서버 응답에 status_url이 없습니다.');
      return;
    }
    const statusUrl = statusUrlRaw.startsWith('http') ? statusUrlRaw : `${window.location.origin}${statusUrlRaw}`;
    pollDetectStatus(statusUrl);
  } catch (e) {
    console.error(e);
    alert('에러 발생: ' + e.message);
  } finally {
    document.getElementById('detectionBtn').disabled = false;
  }
}

function pollDetectStatus(statusUrl) {
  if (panoStatusTimer) clearInterval(panoStatusTimer);
  panoStatusTimer = setInterval(async () => {
    try {
      const res = await fetch(statusUrl);
      const data = await res.json();
      if (!data.success) return;
      if (data.status === 'done' && data.result && data.result.overlay_url) {
        clearInterval(panoStatusTimer);
        panoStatusTimer = null;
        currentOverlayUrl = data.result.overlay_url;
        if (exportBtn) exportBtn.disabled = false;
      } else if (data.status === 'failed') {
        clearInterval(panoStatusTimer);
        panoStatusTimer = null;
        alert('작업 실패: ' + (data.error || 'unknown'));
      }
    } catch (e) {
      console.error(e);
    }
  }, 1200);
}

async function exportOverlay() {
  if (!currentOverlayUrl) {
    alert('Export할 결과가 없습니다.');
    return;
  }
  try {
    const res = await fetch(currentOverlayUrl);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'overlay.png';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
    alert('Export 중 오류가 발생했습니다.');
  }
}
