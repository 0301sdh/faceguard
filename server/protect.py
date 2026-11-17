"""FaceGuard 보호 코드 (Week 8 방식 C').

사진 전체(BGR, OpenCV) → 보호된 사진 전체. 노트북(notebooks/simswap_robustness.ipynb)의
shrink_min → protect_full → pgd_ensemble_eot 를 실험용 코드 없이 그대로 옮겼다.

1. 줄이기: 얼굴이 SimSwap 224 crop 크기가 되도록 사진을 줄이되, 긴 변은 1024 이상 유지
2. 얼굴 찾기: SimSwap과 같은 방식으로 224 정렬 crop + 변환 M
3. 노이즈: SimSwap ArcFace + FaceID ArcFace 를 함께 속이는 EOT PGD (eps 5/255, 변형 4개, steps 30)
4. 붙이기: 노이즈만 M의 역변환으로 사진 좌표에 되돌려 더함

필요한 모델 파일 (models/, git에는 올리지 않음):
- scrfd_10g_bnkps.onnx : 얼굴 찾기 (insightface antelopev2, SimSwap이 쓰는 것)
- simswap_arcface.onnx : SimSwap ArcFace (Colab에서 arcface_checkpoint.tar 를 변환)
- w600k_r50.onnx       : FaceID ArcFace (insightface buffalo_l)
"""
import inspect
import math
import os
import random
import time

import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from insightface.model_zoo import model_zoo
from onnx2torch import convert
from skimage import transform as trans

CROP = 224          # SimSwap crop 크기
MIN_LONG = 1024     # 방식 C': 긴 변 최소값
EPS = 5 / 255
STEPS = 30
N_SAMPLES = 4       # EOT 변형 수 (1개는 변형 없음)
DET_THRESH = 0.6

# SimSwap 정렬 기준 위치 (insightface_func/utils/face_align_ffhqandnewarc.py, 112 기준 5가지 자세)
SIMSWAP_SRC = np.array([
    [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],
    [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],
    [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],
    [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],
    [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]],
], dtype=np.float32)

# insightface(FaceID)가 112×112에서 쓰는 기준 위치: 왼눈, 오른눈, 코, 왼입꼬리, 오른입꼬리
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def pick_device():
    # Mac GPU(mps)는 재 보니 CPU보다 느렸다 (56초 vs 36초, results/week9/local_speed.txt) → 쓰지 않음
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def simswap_matrix(kps, size=CROP):
    # SimSwap estimate_norm(mode='None'): 5가지 자세 중 오차가 가장 작은 변환
    src_all = SIMSWAP_SRC * size / 112
    lmk = np.insert(kps, 2, values=np.ones(5), axis=1)
    best_M, best_err = None, float('inf')
    t = trans.SimilarityTransform()
    for src in src_all:
        t.estimate(kps, src)
        M = t.params[0:2, :]
        err = np.sum(np.sqrt(np.sum((lmk @ M.T - src) ** 2, axis=1)))
        if err < best_err:
            best_M, best_err = M, err
    return best_M


class FaceGuard:
    def __init__(self, model_dir='models', device=None):
        self.device = device or pick_device()
        # 얼굴 찾기 (onnxruntime, CPU)
        self.det = model_zoo.get_model(os.path.join(model_dir, 'scrfd_10g_bnkps.onnx'))
        self._new_api = 'threshold' not in inspect.signature(self.det.detect).parameters
        if self._new_api:     # insightface 0.7 이후: 기준값을 prepare에서 정함
            self.det.prepare(-1, input_size=(640, 640), det_thresh=DET_THRESH)
        else:                 # insightface 0.2 (노트북 환경)
            self.det.prepare(0, input_size=(640, 640))
        # 두 ArcFace (PyTorch로 변환해야 gradient가 흐름)
        self.sw_arc = self._load(os.path.join(model_dir, 'simswap_arcface.onnx'))
        self.fid_arc = self._load(os.path.join(model_dir, 'w600k_r50.onnx'))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _load(self, path):
        m = convert(onnx.load(path)).eval().to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    # ---------- 얼굴 찾기 ----------
    def detect(self, img):
        if self._new_api:
            return self.det.detect(img, max_num=0, metric='default')
        return self.det.detect(img, threshold=DET_THRESH, max_num=0, metric='default')

    def align(self, img):
        # SimSwap app.get(img, 224)과 같음: 첫 번째 얼굴의 224 crop과 변환 M. 못 찾으면 None
        bboxes, kpss = self.detect(img)
        if bboxes.shape[0] == 0:
            return None, None
        M = simswap_matrix(kpss[0])
        return cv2.warpAffine(img, M, (CROP, CROP), borderValue=0.0), M

    def fid_matrix(self, crop_bgr, pad=112):
        # 224 crop에 여백 → 얼굴 찾기 → 5점 → FaceID 112 기준 위치로 옮기는 변환 (crop 좌표 기준)
        padded = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bboxes, kpss = self.detect(padded)
        if bboxes.shape[0] == 0:
            return None
        i = np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
        t = trans.SimilarityTransform()
        t.estimate(kpss[i] - pad, ARCFACE_DST)
        return t.params[:2].astype(np.float32)

    # ---------- 두 모델의 명세서 ----------
    def sw_embed(self, x):
        # SimSwap: 정규화 → 112 (nearest) → ArcFace → 길이 1
        x = (x.contiguous() - self.mean) / self.std
        x = F.interpolate(x, size=(112, 112))
        return F.normalize(self.sw_arc(x), p=2, dim=1)

    def fid_align(self, x, M):
        # cv2.warpAffine(x, M, (112, 112))과 같은 결과를 grid_sample로 (gradient가 흐름)
        Minv = torch.from_numpy(cv2.invertAffineTransform(M)).to(self.device)
        ar = torch.arange(112., device=self.device)
        v, u = torch.meshgrid(ar, ar, indexing='ij')
        src = torch.stack([u, v, torch.ones_like(u)], -1) @ Minv.T
        H, W = x.shape[2:]
        grid = torch.stack([2 * src[..., 0] / (W - 1) - 1, 2 * src[..., 1] / (H - 1) - 1], -1)
        return F.grid_sample(x, grid[None].expand(x.shape[0], -1, -1, -1), mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def fid_embed(self, x, M):
        return F.normalize(self.fid_arc((self.fid_align(x, M) - 0.5) / 0.5), dim=1)

    # ---------- EOT (Week 6) ----------
    def random_theta(self, max_deg=3, max_scale=0.05, max_shift=2, size=CROP):
        angle = math.radians(random.uniform(-max_deg, max_deg))
        scale = 1 + random.uniform(-max_scale, max_scale)
        tx = random.uniform(-max_shift, max_shift) * 2 / size
        ty = random.uniform(-max_shift, max_shift) * 2 / size
        c, s = math.cos(angle) / scale, math.sin(angle) / scale
        return torch.tensor([[c, -s, tx], [s, c, ty]], device=self.device).unsqueeze(0)

    @staticmethod
    def warp(x, theta):
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    # ---------- 노이즈 (Week 7 ensemble) ----------
    def attack(self, face, M_fid, epsilon=EPS, steps=STEPS, n_samples=N_SAMPLES):
        alpha = 2.5 * epsilon / steps
        adv = (face + torch.empty_like(face).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            sims_sw, sims_fid = [], []
            for k in range(n_samples):
                if k == 0:
                    x_adv, x_face = adv, face
                else:
                    theta = self.random_theta()
                    x_adv, x_face = self.warp(adv, theta), self.warp(face, theta)
                with torch.no_grad():
                    orig_sw, orig_fid = self.sw_embed(x_face), self.fid_embed(x_face, M_fid)
                sims_sw.append(F.cosine_similarity(self.sw_embed(x_adv), orig_sw).mean())
                sims_fid.append(F.cosine_similarity(self.fid_embed(x_adv, M_fid), orig_fid).mean())
            loss = sum(sims_sw) / n_samples + sum(sims_fid) / n_samples
            grad, = torch.autograd.grad(loss, adv)
            adv = adv - alpha * grad.sign()
            adv = torch.clamp(face + torch.clamp(adv - face, -epsilon, epsilon), 0, 1)
        return adv.detach()

    # ---------- 전체 ----------
    def shrink(self, photo_bgr):
        # 방식 C': 얼굴이 224 crop 크기가 되게 줄이되 긴 변 MIN_LONG 이상
        _, M = self.align(photo_bgr)
        if M is None:
            return None, None
        s_crop = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        H, W = photo_bgr.shape[:2]
        s = min(1.0, max(s_crop, MIN_LONG / max(H, W)))
        if s >= 1:
            return photo_bgr, 1.0
        return cv2.resize(photo_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA), s

    def to_tensor(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(self.device)

    def protect(self, photo_bgr, seed=0):
        """사진 전체(BGR uint8) → (보호된 사진, 정보). 얼굴을 못 찾으면 (None, 정보).

        seed=None 이면 난수를 다시 정하지 않는다 (노트북처럼 여러 장을 이어서 재현할 때).
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        t0 = time.time()
        small, s = self.shrink(photo_bgr)
        if small is None:
            return None, {'error': 'no_face'}
        crop, M = self.align(small)
        M_fid = self.fid_matrix(crop)
        if M_fid is None:
            return None, {'error': 'no_face'}
        t1 = time.time()

        face = self.to_tensor(crop)
        adv = self.attack(face, M_fid)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

        delta = (adv - face)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1].copy()   # RGB → BGR
        H, W = small.shape[:2]
        delta_big = cv2.warpAffine(delta, cv2.invertAffineTransform(M), (W, H),
                                   flags=cv2.INTER_LINEAR, borderValue=0)
        out = np.clip(small.astype(np.float32) / 255 + delta_big, 0, 1)
        out = (out * 255).round().astype(np.uint8)
        t3 = time.time()
        return out, {'scale': s, 'size': (W, H), 'device': self.device,
                     'seconds': {'detect': t1 - t0, 'noise': t2 - t1, 'paste': t3 - t2, 'total': t3 - t0}}
