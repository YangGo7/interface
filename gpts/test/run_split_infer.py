"""
Test-only script: 여러 YOLO 가중치를 직접 호출해 시각화 이미지를 따로 저장합니다.
API를 거치지 않고 로컬에서 바로 결과를 확인할 때 사용하세요.

Usage 예시:
  python run_split_infer.py --image img.png --out ./test_outputs ^
    --model_teeth yolo11_seg_ver1_800_1024px.pt ^
    --model_caries caries_det.pt ^
    --model_peri periapical.pt ^
    --model_cej cej.pt ^
    --model_bone bonelevel.pt ^
    --model_all <optional>

옵션은 원하는 것만 넣으면 되고, 지정하지 않은 모델은 건너뜁니다.
all.png는 model_all이 지정되면 해당 가중치 단일 추론으로, 지정되지 않으면
teeth/caries/peri/cej/bonelevel 모델들을 순차로 동일 캔버스에 그려서 만듭니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def run_model(img_path: Path, weights: Optional[Path], out_dir: Path, name: str, base: Optional[np.ndarray] = None) -> Optional[Path]:
    if weights is None:
        return None
    model = YOLO(str(weights))
    results = model.predict(str(img_path), verbose=False)
    if not results:
        return None
    # 첫 번째 결과만 사용
    arr = results[0].plot(img=base) if base is not None else results[0].plot()  # numpy array (BGR)
    save_path = out_dir / f"{name}.png"
    cv2.imwrite(str(save_path), arr)
    print(f"[saved] {save_path}")
    return save_path, arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run split inference (local YOLO) and save overlays")
    p.add_argument("--image", required=True, help="path to image/DICOM")
    p.add_argument("--out", default="test_outputs", help="output folder")
    p.add_argument("--model_all", help="(선택) 전체/조합 모델")
    p.add_argument("--model_teeth", help="치식 세그/디텍션 모델 (예: yolo11_seg_ver1_800_1024px.pt)")
    p.add_argument("--model_caries", help="충치 모델 (caries_det.pt 등)")
    p.add_argument("--model_peri", help="치근단염/기타 모델 (periapical.pt 등)")
    p.add_argument("--model_cej", help="CEJ 모델 (cej.pt 등)")
    p.add_argument("--model_bone", help="Bone level 모델 (bonelevel.pt 등)")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def p(v: Optional[str]) -> Optional[Path]:
        return Path(v) if v else None

    # 개별 모델
    run_model(img_path, p(args.model_teeth), out_dir, "teeth")
    run_model(img_path, p(args.model_caries), out_dir, "caries")
    run_model(img_path, p(args.model_peri), out_dir, "peri")

    cej_res = run_model(img_path, p(args.model_cej), out_dir, "cej")
    bone_res = run_model(img_path, p(args.model_bone), out_dir, "bonelevel")

    # extra: cej 우선, 없으면 bonelevel
    if cej_res or bone_res:
        extra_src = cej_res[1] if cej_res else bone_res[1]
        extra_save = out_dir / "extra.png"
        cv2.imwrite(str(extra_save), extra_src)
        print(f"[saved] {extra_save} (extra from {'cej' if cej_res else 'bonelevel'})")

    # all: model_all이 지정되면 단일 실행, 아니면 나머지 모델들을 하나의 캔버스에 누적
    if args.model_all:
        run_model(img_path, p(args.model_all), out_dir, "all")
    else:
        canvas = cv2.imread(str(img_path))
        for w in [p(args.model_teeth), p(args.model_caries), p(args.model_peri), p(args.model_cej), p(args.model_bone)]:
            if w:
                _, canvas = run_model(img_path, w, out_dir, "tmp_all", base=canvas)
        if canvas is not None:
            all_save = out_dir / "all.png"
            cv2.imwrite(str(all_save), canvas)
            print(f"[saved] {all_save} (composed from individual models)")

    print("완료. 지정한 모델만 생성되었으며, 누락된 모델은 건너뛰었습니다.")


if __name__ == "__main__":
    main()
