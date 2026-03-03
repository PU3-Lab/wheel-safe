from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from slope_aggregator import SlopeAggregator

from lib.utils.path import model_path, raw_data_path


class ONNXSlopePipeline:
    """
    ONNX Segmentation -> road mask -> GlobalSlopeAggregator.calculate_slope(... external_mask=mask)
    개선 포인트:
    - letterbox로 종횡비 유지
    - ONNX 출력 shape(NCHW / NHWC / (N,H,W)) 자동 처리
    - external_mask 스케일(0/1 or 0/255) 옵션
    - 디버그 시각화 overlay 저장
    - 경로/예외처리 강화
    """

    def __init__(
        self,
        onnx_path: str | Path,
        config_path: str | Path,
        input_size=(2048, 1024),  # (W, H)
        providers=('CPUExecutionProvider',),
        road_indices=(0),
        obstacle_indices=(11, 12, 13, 14, 15, 16, 17, 18),
        top_cut_ratio=0.40,
        morph_kernel=5,
        mask_scale_255=True,  # GlobalSlopeAggregator가 0/255 마스크를 기대할 가능성이 높아서 기본 True
        debug_dir: str | Path | None = None,
    ):
        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(self.onnx_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

        self.slope_engine = SlopeAggregator(conf_path=str(config_path))

        self.inp_w, self.inp_h = input_size
        self.road_indices = np.array(road_indices, dtype=np.int32)
        self.obstacle_indices = np.array(obstacle_indices, dtype=np.int32)
        self.top_cut_ratio = float(top_cut_ratio)
        self.morph_kernel = int(morph_kernel)
        self.mask_scale_255 = bool(mask_scale_255)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Utils
    # -----------------------------
    @staticmethod
    def _letterbox_rgb(img_rgb: np.ndarray, new_w: int, new_h: int):
        """
        종횡비 유지 resize + padding
        return: padded_img, scale, pad_left, pad_top
        """
        h, w = img_rgb.shape[:2]
        scale = min(new_w / w, new_h / h)
        resized_w = int(round(w * scale))
        resized_h = int(round(h * scale))

        resized = cv2.resize(
            img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros((new_h, new_w, 3), dtype=resized.dtype)
        pad_left = (new_w - resized_w) // 2
        pad_top = (new_h - resized_h) // 2
        canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
        return canvas, scale, pad_left, pad_top, resized_w, resized_h

    @staticmethod
    def _unletterbox_mask(
        mask: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_left: int,
        pad_top: int,
        resized_w: int,
        resized_h: int,
    ):
        """
        letterbox된 mask를 원본 해상도로 복원
        mask: (new_h,new_w)
        """
        # 패딩 제거
        cropped = mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
        # 원본으로 resize
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _normalize_chw(img_rgb_float01: np.ndarray, mean: np.ndarray, std: np.ndarray):
        x = (img_rgb_float01 - mean) / std
        x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1,3,H,W)
        return x

    @staticmethod
    def _infer_to_classmap(outputs0: np.ndarray):
        """
        outputs[0] -> class map (H,W)로 변환
        지원:
        - (N,C,H,W) logits
        - (N,H,W,C) logits
        - (N,H,W) already class index
        - (H,W) already class index
        """
        out = outputs0
        if out.ndim == 4:
            # NCHW or NHWC
            if out.shape[1] <= 256:  # 보통 C가 1~200대, H/W는 300~2000대
                # (N,C,H,W)
                cls = np.argmax(out, axis=1)
                return cls[0].astype(np.int32)
            else:
                # (N,H,W,C)
                cls = np.argmax(out, axis=-1)
                return cls[0].astype(np.int32)

        if out.ndim == 3:
            # (N,H,W) or (H,W,C)
            if out.shape[0] == 1:
                return out[0].astype(np.int32)
            # 애매한 경우: (H,W,C)면 argmax 필요
            # out.shape[-1]이 클래스 수처럼 작으면 처리
            if out.shape[-1] <= 256:
                return np.argmax(out, axis=-1).astype(np.int32)
            # 그 외는 (N,H,W)로 간주
            return out[0].astype(np.int32)

        if out.ndim == 2:
            return out.astype(np.int32)

        raise ValueError(f'Unsupported output shape: {out.shape}')

    # -----------------------------
    # Core
    # -----------------------------
    def get_road_mask(self, img_bgr: np.ndarray):
        """
        return:
        - road_mask_uint8: (H,W) 0/255 or 0/1 (설정에 따름)
        - pred_full: (H,W) class index map
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError('Empty image')

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) letterbox (종횡비 유지)
        padded, scale, pad_left, pad_top, resized_w, resized_h = self._letterbox_rgb(
            img_rgb, self.inp_w, self.inp_h
        )

        # 2) normalize + infer
        x = padded.astype(np.float32) / 255.0
        x = self._normalize_chw(x, self.mean, self.std)

        outputs = self.session.run(None, {self.input_name: x})
        pred_lb = self._infer_to_classmap(outputs[0])  # (inp_h, inp_w)

        # 3) unletterbox -> 원본 크기
        pred_full = self._unletterbox_mask(
            pred_lb,
            orig_w=w,
            orig_h=h,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            resized_w=resized_w,
            resized_h=resized_h,
        )

        # 4) class filtering
        # NOTE: obstacle_indices가 road_indices와 겹치지 않으면 ~obstacle은 큰 의미 없음.
        # 그래도 안전하게 유지.
        # is_road = np.isin(pred_full, self.road_indices)
        # is_obst = np.isin(pred_full, self.obstacle_indices)
        # road_mask = (is_road & ~is_obst).astype(np.uint8)
        road_mask = np.isin(pred_full, self.road_indices).astype(np.uint8)

        # 5) top cut
        # cut = int(h * self.top_cut_ratio)
        # if 0 < cut < h:
        #     road_mask[:cut, :] = 0

        # 6) morphology open
        if self.morph_kernel >= 3:
            k = (
                self.morph_kernel
                if self.morph_kernel % 2 == 1
                else self.morph_kernel + 1
            )
            kernel = np.ones((k, k), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # 7) scale for external_mask
        if self.mask_scale_255:
            road_mask = (road_mask * 255).astype(np.uint8)

        return road_mask, pred_full

    def _save_debug(
        self,
        stem: str,
        img_bgr: np.ndarray,
        road_mask: np.ndarray,
        pred_full: np.ndarray,
    ):
        if not self.debug_dir:
            return

        # overlay
        overlay = img_bgr.copy()
        # road_mask가 0/255면 그대로, 0/1이면 *255
        m = road_mask
        if m.max() == 1:
            m = (m * 255).astype(np.uint8)

        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        alpha = 0.35
        overlay[m > 0] = (overlay[m > 0] * (1 - alpha) + green[m > 0] * alpha).astype(
            np.uint8
        )

        cv2.imwrite(str(self.debug_dir / f'{stem}_roadmask.png'), m)
        cv2.imwrite(str(self.debug_dir / f'{stem}_overlay.png'), overlay)

        # pred_full을 보기 좋게 컬러맵으로 저장(클래스 id 시각화)
        pred_vis = (
            pred_full.astype(np.float32) / max(1, pred_full.max()) * 255.0
        ).astype(np.uint8)
        pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(self.debug_dir / f'{stem}_pred.png'), pred_vis)

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
            raise FileNotFoundError(f'confidence not found: {conf_path}')

        img = cv2.imread(str(left_path))
        if img is None:
            raise RuntimeError(f'Failed to read image: {left_path}')

        road_mask, pred_full = self.get_road_mask(img)

        # 디버그 저장
        self._save_debug(left_path.stem, img, road_mask, pred_full)

        # slope 계산
        result = self.slope_engine.calculate_slope(
            disp_path,
            conf_path,
            external_mask=None,
        )
        print(result['avg_slope'])

        avg = result.get('avg_slope', None)
        if avg is not None:
            print(f'분석 결과 - 평균 경사도: {avg:.2f}°')
        else:
            print(
                '분석 결과 - avg_slope 키가 result에 없음. result keys:',
                list(result.keys()),
            )

        return result


if __name__ == '__main__':
    depth_nums = [('001', 'ZED1_KSC_001032'), ('007', 'ZED4_KSC_010545')]

    for _, num in enumerate(depth_nums):
        depth_path = raw_data_path(num[0])

        pipe = ONNXSlopePipeline(
            onnx_path=model_path() / 'pidnet.onnx',
            config_path=depth_path / f'Depth_{num[0]}.conf',
            debug_dir='debug_out',  # 여기로 overlay/pred/mask 저장됨
            mask_scale_255=False,  # 필요하면 False로
            top_cut_ratio=0.40,
            morph_kernel=5,
        )

        prefix = num[1]

        res = pipe.run_pipeline(
            left_path=depth_path / f'{prefix}_left.png',
            disp_path=depth_path / f'{prefix}_disp16.png',
            conf_path=depth_path / f'{prefix}_confidence_save.png',
        )
        print(res)
