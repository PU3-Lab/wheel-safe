from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from gpt.debug_heatmap import plot_debug_figure
from gpt.slope_aggregator import SlopeAggregator
from lib.utils.path import output_path


class ONNXSlopePipeline:
    def __init__(
        self,
        *,
        onnx_path: str | Path,
        config_path: str | Path,  # Depth_*.conf
        mode: str = 'LEFT_CAM_FHD',
        morph_kernel: int = 7,
        # slope params
        disp_scale: float = 16.0,
        conf_th: int = 10,
        min_d: float = 0.8,
        max_d: float = 5.0,
        roi_ratio: float = 0.55,
        ransac_iters: int = 500,
        ransac_inlier_thresh_m: float = 0.02,
        max_points: int = 50000,
        debug_print: bool = True,
    ):
        self.onnx_path = Path(onnx_path)
        self.config_path = Path(config_path)
        self.mode = mode
        self.morph_kernel = int(morph_kernel)

        self.session = ort.InferenceSession(str(self.onnx_path))

        # ✅ slope engine 생성
        self.slope_engine = SlopeAggregator(
            config_path=self.config_path,
            mode=self.mode,
            disp_scale=disp_scale,
            conf_th=conf_th,
            min_d=min_d,
            max_d=max_d,
            roi_ratio=roi_ratio,
            ransac_iters=ransac_iters,
            ransac_inlier_thresh_m=ransac_inlier_thresh_m,
            max_points=max_points,
            debug_print=debug_print,
        )

        # TODO: 네 기존 ONNX 세션 초기화 코드가 있으면 여기 유지
        # self.session = ...

    def get_road_mask(self, img_bgr):
        """
        return:
        - road_mask: (H,W) uint8 (0/255)
        - pred_full: (H,W) int class map
        """
        h, w = img_bgr.shape[:2]

        # 1) 모델 입력 shape 읽기 (대부분 NCHW)
        inp = self.session.get_inputs()[0]
        in_name = inp.name
        in_shape = inp.shape  # e.g. [1, 3, 1024, 2048] or [1,3,'H','W']

        # 동적 shape일 수도 있으니 기본값을 잡아줌
        # (네 에러 기준: 1024x2048이 정답)
        target_h, target_w = 1024, 2048
        if (
            len(in_shape) == 4
            and isinstance(in_shape[2], int)
            and isinstance(in_shape[3], int)
        ):
            target_h, target_w = int(in_shape[2]), int(in_shape[3])

        # 2) 모델 입력 크기로 resize
        resized = cv2.resize(
            img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # 3) 전처리 (너 기존 방식 유지: 0~1 정규화, CHW)
        x = resized.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None, :, :, :]  # (1,3,H,W)

        # 4) 추론
        out = self.session.run(None, {in_name: x})[0]  # 보통 (1,C,H,W)
        pred = out.argmax(1)[0].astype(np.uint8)  # (H,W)

        # 5) 원본 해상도로 되돌리기
        pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # 6) road_mask 생성 (너 설정대로 road_indices=(0)라면)
        # road_mask = (pred_full == 0).astype(np.uint8) * 255
        road_mask = np.isin(pred_full, [0, 1]).astype(np.uint8) * 255

        kernel = np.ones((11, 11), np.uint8)
        road_mask = cv2.erode(road_mask, kernel, iterations=1)

        kernel = np.ones((9, 9), np.uint8)

        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        h, w = road_mask.shape

        road_mask[:, : int(w * 0.2)] = 0
        road_mask[:, int(w * 0.8) :] = 0

        print(f'road_mask : {road_mask}')

        # 7) morphology (기존 유지)
        if getattr(self, 'morph_kernel', 0) and self.morph_kernel > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_kernel, self.morph_kernel),
            )
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k)

        return road_mask, pred_full

    def _save_debug(
        self,
        stem: str,
        img: np.ndarray,
        road_mask: np.ndarray,
        pred_full: np.ndarray,
    ):
        debug_dir = Path('debug')
        debug_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(debug_dir / f'{stem}_input.png'),
            img,
        )

        cv2.imwrite(
            str(debug_dir / f'{stem}_road_mask.png'),
            road_mask,
        )

        vis = (pred_full * 10).astype(np.uint8)

        cv2.imwrite(
            str(debug_dir / f'{stem}_seg.png'),
            vis,
        )

    def run_pipeline(
        self, left_path: str | Path, disp_path: str | Path, conf_path: str | Path
    ):
        left_path = Path(left_path)
        disp_path = Path(disp_path)
        conf_path = Path(conf_path)

        if not left_path.exists():
            raise FileNotFoundError(f'left image not found: {left_path}')
        if not disp_path.exists():
            raise FileNotFoundError(f'disparity not found: {disp_path}')
        if not conf_path.exists():
            raise FileNotFoundError(f'confidence image not found: {conf_path}')

        img = cv2.imread(str(left_path))
        if img is None:
            raise RuntimeError(f'Failed to read image: {left_path}')

        road_mask, pred_full = self.get_road_mask(img)

        # debug 저장
        self._save_debug(left_path.stem, img, road_mask, pred_full)

        # ✅ external_mask 적용 (핵심)
        external_mask = (road_mask > 0).astype(np.uint8)

        result = self.slope_engine.calculate_slope(
            disp_path=disp_path,
            conf_img_path=conf_path,
            external_mask=external_mask,
        )

        avg_slope = result.get('avg_slope', float('nan'))
        print('avg_slope:', avg_slope, 'reason:', result.get('reason'))

        # 너 기존 plot_debug_figure / output 저장 로직이 있으면 여기서 그대로 사용
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f'{left_path.stem}_slope_{avg_slope:.2f}deg_{timestamp}.png'
        out_path = output_path() / out_name
        plot_debug_figure(
            img_bgr=img, pred_full=pred_full, result=result, out_path=out_path
        )

        return result
