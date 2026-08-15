#!/usr/bin/env python3
"""PitchHomography — sabit kamera için saha homografisi (manuel kalibrasyon yolu).

Bu modül A'nın omurgasıdır: maç başına BİR KEZ yapılan manuel (4..10 tık)
kalibrasyon, lens distorsiyonu düzeltme, hot-path'te seyrek ayak-noktası
projeksiyonu, mekânsal (per-zone) reprojection QA ve kamera başına JSON kilidi.

Tasarım gerçeği: kamera SABİT olduğu için H tüm maç boyunca değişmez. Tespit HAM
(distorted) frame üzerinde çalışır; yalnızca seyrek ayak noktaları
cv2.undistortPoints ile düzeltilip H ile saha-metresine taşınır — sıcak yolda
tam-frame CV yoktur, GPU yoktur.

Homografi yönü (sabit): H_img2pitch, UNDISTORTED piksel -> saha metresi.
    [X, Y, w]^T = H_img2pitch @ [u_und, v_und, 1]^T,  sonra X/w, Y/w.
H_pitch2img = inv(H_img2pitch) (QA overlay + template reprojeksiyonu için).

Lisans: yalnızca OpenCV (BSD) + numpy. GPU yok.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # template.py paralel yazılıyor; import-time bağımlılık yaratma
    from pitch.template import PitchTemplate

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_pts(arr) -> np.ndarray:
    """(N,2) float64 piksel/metre dizisine normalize et."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1 and a.size == 2:
        a = a.reshape(1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"nokta dizisi (N,2) olmalı, alınan {a.shape}")
    return a


class PitchHomography:
    """Bir kamera için kalıcı saha homografisi + distorsiyon + QA.

    Kullanım (manuel yol):
        homo = PitchHomography("cankaya_cam2", PitchTemplate.five_a_side())
        homo.set_distortion(K, dist)               # opsiyonel ama önerilir
        qa = homo.calibrate_manual(img_pts, world_pts)
        homo.save("calib/cankaya_cam2.json")
        ...
        pitch_xy = homo.pixel_to_pitch(foot_pts_px)  # sıcak yol, vektörize
    """

    # -------------------------------------------------------------- init ---
    def __init__(self, camera_id: str, template: "PitchTemplate | None" = None):
        self.camera_id = camera_id
        self.template = template
        # lens distorsiyonu
        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.undistort_applied: bool = False
        # homografi
        self.H_img2pitch: np.ndarray | None = None
        self.H_pitch2img: np.ndarray | None = None
        # meta
        self.status: str = "uncalibrated"   # uncalibrated|manual|auto_accepted|rejected|recalibrate
        self.calib_method: str | None = None  # manual | auto_chamfer
        self.orientation_hint: str | None = None
        self.scale_anchor: dict | None = None
        self.calib_date: str | None = None
        self.source_clip: str | None = None
        self._qa: dict | None = None
        # kalibrasyon karşılıkları (QA / reprojection için saklanır, UNDISTORTED px)
        self._img_pts_und: np.ndarray | None = None
        self._world_pts: np.ndarray | None = None

    # ---------------------------------------------------------- distortion --
    def set_distortion(self, K: np.ndarray, dist: np.ndarray) -> None:
        """Lens iç parametrelerini ata (kamera başına bir kez).

        K: (3,3) kamera matrisi, dist: (5,) [k1,k2,p1,p2,k3].
        """
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.undistort_applied = True

    @classmethod
    def estimate_distortion_checkerboard(cls, frames, pattern_size, square_m):
        """Satranç tahtası ile K, dist kestirimi (en güvenilir yöntem).

        frames: BGR görüntü listesi; pattern_size: (cols, rows) iç köşe sayısı;
        square_m: kare kenarı (metre). Döner: (K, dist).
        """
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= float(square_m)
        objpoints, imgpoints = [], []
        img_shape = None
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if not ok:
                continue
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            objpoints.append(objp)
            imgpoints.append(corners)
        if len(objpoints) < 3:
            raise RuntimeError(
                f"yetersiz satranç tespiti ({len(objpoints)}); >=3 frame gerek")
        _, K, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)
        return K, dist.reshape(-1)

    @classmethod
    def estimate_distortion_from_lines(cls, img_shape, bowed_polylines_px):
        """Bilinen-düz saha çizgilerini düzleştirerek tek-parametreli k1 kestir.

        Bukhari-Dailey bölme-modeli (division model) basitleştirmesi: gerçekte
        düz olan çizgilerin görüntüdeki eğriliğini en aza indiren tek radyal
        katsayıyı arar.

        DÜRÜST UYARI: satranç tahtasından ZAYIFtır (yalnız k1, ilkesel olarak
        merkez/odak varsayımlı). Bu yöntem kullanıldığında düşük-güven
        bayrağı (status/qa) ile işaretlenmelidir.

        img_shape: (H, W); bowed_polylines_px: her biri (>=3, 2) olan, gerçekte
        düz olduğu bilinen çizgi-noktaları listesi. Döner: (K, dist).
        """
        H, W = int(img_shape[0]), int(img_shape[1])
        cx, cy = W / 2.0, H / 2.0
        f0 = float(max(W, H))  # makul başlangıç odak (yalnız undistort için)
        K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]], dtype=np.float64)

        polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                 for p in bowed_polylines_px if len(p) >= 3]
        if not polys:
            raise ValueError("en az bir >=3-noktalı düz çizgi gerek")
        norm = (max(W, H) / 2.0) ** 2  # k1 normalizasyon ölçeği

        def straighten(pts, k1):
            # bölme-modeli ters-eşlem: r_u = r_d / (1 + k1 * r_d^2)
            d = pts - np.array([cx, cy])
            r2 = (d ** 2).sum(axis=1) / norm
            scale = 1.0 / (1.0 + k1 * r2)
            return pts * 0 + np.array([cx, cy]) + d * scale[:, None]

        def line_residual(pts):
            # noktaların en-iyi-uyan doğruya dik mesafe RMS'i
            c = pts.mean(axis=0)
            u, s, vt = np.linalg.svd(pts - c)
            normal = vt[-1]  # en küçük tekil yöne karşılık gelen normal
            d = (pts - c) @ normal
            return float(np.sqrt((d ** 2).mean()))

        def total_cost(k1):
            return sum(line_residual(straighten(p, k1)) for p in polys)

        # 1B kaba+ince tarama (scipy bağımlılığı eklemeden, BSD-temiz)
        best_k1, best_c = 0.0, total_cost(0.0)
        for k1 in np.linspace(-1.0, 1.0, 201):
            c = total_cost(float(k1))
            if c < best_c:
                best_c, best_k1 = c, float(k1)
        lo, hi = best_k1 - 0.01, best_k1 + 0.01
        for k1 in np.linspace(lo, hi, 201):
            c = total_cost(float(k1))
            if c < best_c:
                best_c, best_k1 = c, float(k1)

        # bölme-modeli k1'ini Brown-Conrady k1'e yaklaşık çevir (işaret/ölçek)
        k1_brown = best_k1 / norm
        dist = np.array([k1_brown, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return K, dist

    # --------------------------------------------------------- calibration --
    def calibrate_manual(self, img_pts_px: np.ndarray, world_pts_m: np.ndarray,
                         ransac_thresh_px: float = 5.0,
                         already_undistorted: bool = False) -> dict:
        """Manuel kalibrasyon: HAM-piksel tıklamaları + saha-metre karşılıkları.

        img_pts_px: (N>=4, 2) HAM (distorted) piksel tıklamaları; dahili olarak
        cv2.undistortPoints ile düzeltilir. world_pts_m: (N, 2) saha metresi.
        H = cv2.findHomography(undist_img_pts, world_pts, RANSAC, thresh) (DLT).

        already_undistorted=True ise img_pts ZATEN undistorted piksel uzayında
        kabul edilir (pixel_to_pitch ile aynı konvansiyon; line_calib bunu kullanır).
        Varsayılan False = davranış değişmez.

        Döner: reproject_error_map() QA sözlüğü. status="manual" yapar.
        """
        img = _as_pts(img_pts_px)
        world = _as_pts(world_pts_m)
        if len(img) != len(world):
            raise ValueError("img_pts ve world_pts eşit uzunlukta olmalı")
        if len(img) < 4:
            raise ValueError("homografi için >=4 nokta gerek")

        img_und = img if already_undistorted else self.undistort_points(img)
        H, mask = cv2.findHomography(
            img_und.reshape(-1, 1, 2), world.reshape(-1, 1, 2),
            cv2.RANSAC, float(ransac_thresh_px))
        if H is None:
            raise RuntimeError("findHomography çözülemedi (dejenere noktalar?)")

        self.H_img2pitch = np.asarray(H, dtype=np.float64)
        self.H_pitch2img = np.linalg.inv(self.H_img2pitch)
        self._img_pts_und = img_und
        self._world_pts = world
        self.status = "manual"
        self.calib_method = "manual"
        self.calib_date = _utc_now_iso()

        qa = self.reproject_error_map()
        self._qa = qa
        return qa

    def set_homography(self, H_img2pitch: np.ndarray, status: str) -> None:
        """auto_calib'in kendi H'sini push etmesi için (chamfer sonrası)."""
        self.H_img2pitch = np.asarray(H_img2pitch, dtype=np.float64).reshape(3, 3)
        self.H_pitch2img = np.linalg.inv(self.H_img2pitch)
        self.status = status
        self.calib_method = "auto_chamfer"
        self.calib_date = _utc_now_iso()

    # ---------------------------------------------------------------- apply -
    def _require_H(self):
        if self.H_img2pitch is None:
            raise RuntimeError("homografi kurulmadı (önce calibrate_manual/"
                               "set_homography)")

    def undistort_points(self, pts_px: np.ndarray) -> np.ndarray:
        """HAM (N,2) piksel -> distorsiyonu düzeltilmiş (N,2) piksel.

        Distorsiyon atanmamışsa noktaları değiştirmeden döndürür (kimlik).
        cv2.undistortPoints(..., P=K) çıkışı yine piksel uzayındadır.
        """
        pts = _as_pts(pts_px)
        if self.K is None or self.dist is None:
            return pts.copy()
        out = cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float64),
            self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def pixel_to_pitch(self, pts_px: np.ndarray,
                       already_undistorted: bool = False) -> np.ndarray:
        """Ayak-noktası pikselleri -> saha metresi (N,2). Sıcak yol, vektörize.

        already_undistorted=False ise önce undistortPoints uygulanır.
        """
        self._require_H()
        pts = _as_pts(pts_px)
        und = pts if already_undistorted else self.undistort_points(pts)
        out = cv2.perspectiveTransform(
            und.reshape(-1, 1, 2).astype(np.float64), self.H_img2pitch)
        return out.reshape(-1, 2)

    def pitch_to_pixel(self, pts_m: np.ndarray,
                       distorted: bool = False) -> np.ndarray:
        """Saha metresi -> piksel (N,2).

        distorted=False (default, geriye-uyumlu): UNDISTORTED piksel; ham frame'e
          bindirme icin goruntu undistort edilmeli (qa_overlay bunu yapar).
        distorted=True: HAM (distorted) piksel = pixel_to_pitch'in TAM TERSI. Ham foot
          pikselleriyle ayni uzayda kiyas (RULE-5 polygon, raw-overlay) icin -> pitch<->pixel
          distorsiyon-konvansiyonu TUTARLI olur (cift-undistort footgun'i biter).
        """
        self._require_H()
        pts = _as_pts(pts_m)
        und = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float64), self.H_pitch2img).reshape(-1, 2)
        if not distorted or not getattr(self, "undistort_applied", False):
            return und
        # UNDISTORTED px -> HAM px (ileri Brown-Conrady): K^-1 normalize -> projectPoints
        Ki = np.linalg.inv(self.K)
        nrm = (Ki @ np.c_[und, np.ones(len(und))].T).T[:, :2]
        obj = np.c_[nrm, np.ones(len(und))].astype(np.float64)
        raw, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3),
                                   self.K, self.dist.reshape(-1))
        return raw.reshape(-1, 2)

    def in_pitch(self, pts_m: np.ndarray, margin_m: float = 2.0) -> np.ndarray:
        """(N,) bool: nokta saha dikdörtgeni + margin içinde mi.

        Tribün/file/yansıma FP'lerini eler. Template yoksa allowed-range'den
        makul bir dikdörtgen kullanır.
        """
        pts = _as_pts(pts_m)
        L, W = self._dims_m()
        m = float(margin_m)
        x, y = pts[:, 0], pts[:, 1]
        return ((x >= -m) & (x <= L + m) & (y >= -m) & (y <= W + m))

    # ------------------------------------------------------------------ QA --
    def reprojection_error(self) -> float:
        """Skaler özet: kalibrasyon noktalarının medyan reprojection hatası (px).

        Saklanan karşılıklar (undistorted px <-> metre) üzerinden hesaplanır.
        QA'nın TEK-SAYI özetidir; mekânsal dağılım için reproject_error_map()
        kullanın (grazing açıda tek sayı yanıltıcıdır).
        """
        if self._img_pts_und is None or self._world_pts is None:
            return float("nan")
        proj = self.pitch_to_pixel(self._world_pts)
        err = np.linalg.norm(proj - self._img_pts_und, axis=1)
        return float(np.median(err))

    def reproject_error_map(self, line_pixels_px: np.ndarray | None = None) -> dict:
        """Mekânsal QA — asla tek sayı değil.

        Kalibrasyon karşılıklarını UNDISTORTED görüntüye geri-projekte eder ve
        hatayı derinlik (v) bölgelerine göre ayırır. Hata, landmark dışbükey
        kabuğunun dışında ve ufka yakın (grazing açı) büyür.

        Döner: {median_px, p95_px, per_zone:{near,mid,far}, n_landmarks,
                both_halves, outside_hull_flag, horizon_warn}
        """
        out = dict(median_px=float("nan"), p95_px=float("nan"),
                   per_zone=dict(near=float("nan"), mid=float("nan"),
                                 far=float("nan")),
                   n_landmarks=0, both_halves=False,
                   outside_hull_flag=False, horizon_warn=False)
        if self._img_pts_und is None or self._world_pts is None:
            return out

        img_und = self._img_pts_und
        proj = self.pitch_to_pixel(self._world_pts)
        err = np.linalg.norm(proj - img_und, axis=1)
        out["n_landmarks"] = int(len(err))
        out["median_px"] = float(np.median(err))
        out["p95_px"] = float(np.percentile(err, 95))

        # derinlik bölgeleri: görüntü v koordinatına göre (üst=uzak, alt=yakın)
        v = img_und[:, 1]
        vmin, vmax = float(v.min()), float(v.max())
        if vmax > vmin:
            t = (v - vmin) / (vmax - vmin)  # 0=üst/uzak .. 1=alt/yakın
        else:
            t = np.zeros_like(v)
        far_m = t < 1 / 3
        mid_m = (t >= 1 / 3) & (t < 2 / 3)
        near_m = t >= 2 / 3

        def _z(m):
            return float(np.median(err[m])) if m.any() else float("nan")
        out["per_zone"] = dict(near=_z(near_m), mid=_z(mid_m), far=_z(far_m))

        # her iki saha yarısı temsil ediliyor mu (L/R simetri çözümü için kritik)
        L, _ = self._dims_m()
        X = self._world_pts[:, 0]
        out["both_halves"] = bool((X < L / 2).any() and (X > L / 2).any())

        # ek line_pixels verildiyse dışbükey-kabuk dışı bayrağı
        if line_pixels_px is not None and len(img_und) >= 3:
            hull = cv2.convexHull(img_und.astype(np.float32))
            lp = _as_pts(line_pixels_px).astype(np.float32)
            outside = sum(
                cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) < 0
                for p in lp)
            out["outside_hull_flag"] = bool(outside > 0)

        # ufuk uyarısı: uzak bölge hatası yakın bölgenin >3 katıysa grazing açı
        nz, fz = out["per_zone"]["near"], out["per_zone"]["far"]
        if not np.isnan(nz) and not np.isnan(fz) and nz > 0 and fz > 3 * nz:
            out["horizon_warn"] = True
        return out

    def qa_overlay(self, img_bgr: np.ndarray) -> np.ndarray:
        """Reprojekte edilmiş template çizgilerini görüntü üzerine çiz.

        Distorsiyon atanmışsa görüntüyü önce undistort eder ki çizgiler hizalı
        olsun (pitch_to_pixel undistorted uzayda döner).
        """
        self._require_H()
        img = img_bgr.copy()
        if self.K is not None and self.dist is not None:
            img = cv2.undistort(img, self.K, self.dist)

        # template çizgilerini çiz
        segs = self._template_segments_m()
        for (p0, p1) in segs:
            a = self.pitch_to_pixel(np.array([p0]))[0]
            b = self.pitch_to_pixel(np.array([p1]))[0]
            cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))), (0, 255, 0), 2)

        # kalibrasyon noktaları: tıklanan (sarı) vs reprojekte (kırmızı)
        if self._img_pts_und is not None and self._world_pts is not None:
            proj = self.pitch_to_pixel(self._world_pts)
            for (ix, iy), (px, py) in zip(self._img_pts_und, proj):
                cv2.circle(img, (int(round(ix)), int(round(iy))), 5,
                           (0, 255, 255), -1)
                cv2.circle(img, (int(round(px)), int(round(py))), 5,
                           (0, 0, 255), 2)
        return img

    # --------------------------------------------------- drift / vibration --
    def drift_check(self, line_map: np.ndarray, max_resid_px: float) -> bool:
        """Kilitli H, mevcut maçın statik çizgi haritasıyla hâlâ uyumlu mu.

        Template çizgilerini reprojekte eder ve line_map (çizgi olasılığı,
        HxW float [0,1] veya ikili) üzerindeki distance-transform'a göre medyan
        artığı ölçer. Artık max_resid_px'i aşarsa True => RECALIBRATE gerek.
        Statik kamera "sabit"tir ama sonsuza dek değil (sehpa kayması, darbe).
        """
        self._require_H()
        lm = np.asarray(line_map)
        if lm.dtype != np.uint8 and lm.max() <= 1.0:
            binary = (lm > 0.5).astype(np.uint8)
        else:
            binary = (lm > 0).astype(np.uint8)
        # çizgi pikseli=1 -> distance_transform için tersini al
        dt = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)

        pts = self._template_line_points_m()
        if len(pts) == 0:
            return False
        px = self.pitch_to_pixel(pts)
        h, w = binary.shape[:2]
        u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        resid = float(np.median(dt[v, u]))
        return resid > float(max_resid_px)

    def stitch_motion_gate_m(self, p0_m, p1_m, dt_s,
                             max_speed_mps: float = 12.0) -> bool:
        """Metre-domeninde tracklet birleştirme kapısı.

        İki tracklet ucu saha koordinatında dt_s içinde max_speed_mps'i aşan bir
        hız gerektiriyorsa birleştirilemez (statik-kamera bedava sinyali;
        broadcast yöntemlerinde yoktur). Döner: True => birleştirmeye İZİN VAR.
        """
        a = np.asarray(p0_m, dtype=np.float64).reshape(2)
        b = np.asarray(p1_m, dtype=np.float64).reshape(2)
        dt = float(dt_s)
        if dt <= 0:
            return False
        speed = float(np.linalg.norm(b - a) / dt)
        return speed <= float(max_speed_mps)

    # ----------------------------------------------------------- persist ----
    def save(self, path: str) -> None:
        """Kalibrasyonu JSON şemasına yaz (calib/<camera_id>.json)."""
        self._require_H()
        L, W = self._dims_m()
        data = {
            "schema_version": SCHEMA_VERSION,
            "camera_id": self.camera_id,
            "pitch_dims_m": {"L": L, "W": W},
            "template": self._template_to_dict(),
            "K": self.K.tolist() if self.K is not None else None,
            "dist": self.dist.tolist() if self.dist is not None else None,
            "undistort_applied": bool(self.undistort_applied),
            "H_img2pitch": self.H_img2pitch.tolist(),
            "H_pitch2img": self.H_pitch2img.tolist(),
            "scale_anchor": self.scale_anchor,
            "qa": self._qa if self._qa is not None else self.reproject_error_map(),
            "status": self.status,
            "calib_method": self.calib_method,
            "orientation_hint": self.orientation_hint,
            "calib_date": self.calib_date or _utc_now_iso(),
            "source_clip": self.source_clip,
            # QA/yeniden-değerlendirme için karşılıklar (undistorted px <-> metre)
            "_img_pts_und": (self._img_pts_und.tolist()
                             if self._img_pts_und is not None else None),
            "_world_pts": (self._world_pts.tolist()
                           if self._world_pts is not None else None),
        }
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PitchHomography":
        """Kalıcı JSON'dan PitchHomography geri yükle."""
        with open(path) as f:
            d = json.load(f)
        template = cls._template_from_dict(d.get("template"))
        homo = cls(d["camera_id"], template)
        if d.get("K") is not None and d.get("dist") is not None:
            homo.set_distortion(np.array(d["K"]), np.array(d["dist"]))
            homo.undistort_applied = bool(d.get("undistort_applied", True))
        homo.H_img2pitch = np.array(d["H_img2pitch"], dtype=np.float64)
        homo.H_pitch2img = np.array(d["H_pitch2img"], dtype=np.float64)
        homo.status = d.get("status", "manual")
        homo.calib_method = d.get("calib_method")
        homo.orientation_hint = d.get("orientation_hint")
        homo.scale_anchor = d.get("scale_anchor")
        homo.calib_date = d.get("calib_date")
        homo.source_clip = d.get("source_clip")
        homo._qa = d.get("qa")
        if d.get("_img_pts_und") is not None:
            homo._img_pts_und = np.array(d["_img_pts_und"], dtype=np.float64)
        if d.get("_world_pts") is not None:
            homo._world_pts = np.array(d["_world_pts"], dtype=np.float64)
        # template yoksa dims'i JSON'dan tut
        homo._fallback_dims = (d["pitch_dims_m"]["L"], d["pitch_dims_m"]["W"])
        return homo

    # ----------------------------------------------- annotate CLI helper ----
    @staticmethod
    def annotate_points(frame_path: str) -> np.ndarray:
        """Bir frame üzerinde 4+ nokta tıklamak için basit cv2 mouse CLI.

        Sol tık: nokta ekle. 'u': son noktayı geri al. 'q'/Enter: bitir.
        Döner: (N,2) HAM piksel tıklamaları. Her tıklamanın saha-metre
        karşılığı operatör tarafından ayrıca girilir (bkz. __main__).
        """
        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(frame_path)
        pts: list[tuple[float, float]] = []
        win = "calib: sol-tik nokta, 'u' geri-al, 'q'/Enter bitir"

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((float(x), float(y)))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            disp = img.copy()
            for i, (px, py) in enumerate(pts):
                cv2.circle(disp, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i), (int(px) + 6, int(py) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 13):  # q veya Enter
                break
            if k == ord("u") and pts:
                pts.pop()
        cv2.destroyAllWindows()
        return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))

    # ----------------------------------------------- template yardımcıları --
    def _dims_m(self) -> tuple[float, float]:
        """Saha (L, W) metre — template'ten, yoksa fallback/varsayılan."""
        if self.template is not None and hasattr(self.template, "dims_m"):
            return float(self.template.dims_m[0]), float(self.template.dims_m[1])
        if hasattr(self, "_fallback_dims"):
            return float(self._fallback_dims[0]), float(self._fallback_dims[1])
        return 28.0, 18.0  # 5v5 makul varsayılan (yalnız in_pitch sınırı için)

    def _template_segments_m(self):
        """Template çizgi segmentleri; yoksa saha dikdörtgeni kenarları."""
        if self.template is not None and hasattr(self.template, "line_segments_m"):
            try:
                segs = self.template.line_segments_m
                if segs:
                    return [(tuple(p0), tuple(p1)) for (p0, p1) in segs]
            except Exception:
                pass
        L, W = self._dims_m()
        return [((0, 0), (L, 0)), ((L, 0), (L, W)),
                ((L, W), (0, W)), ((0, W), (0, 0)),
                ((L / 2, 0), (L / 2, W))]

    def _template_line_points_m(self) -> np.ndarray:
        """Chamfer/drift için yoğun örneklenmiş çizgi noktaları (M,2)."""
        if self.template is not None and hasattr(self.template, "line_points_m"):
            try:
                return np.asarray(self.template.line_points_m(), dtype=np.float64)
            except Exception:
                pass
        # fallback: segmentleri 0.25 m adımla örnekle
        pts = []
        for (p0, p1) in self._template_segments_m():
            p0 = np.asarray(p0, float)
            p1 = np.asarray(p1, float)
            d = np.linalg.norm(p1 - p0)
            n = max(2, int(d / 0.25))
            for t in np.linspace(0, 1, n):
                pts.append(p0 + t * (p1 - p0))
        return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2))

    def _template_to_dict(self):
        if self.template is None:
            return None
        if hasattr(self.template, "to_dict"):
            try:
                return self.template.to_dict()
            except Exception:
                pass
        # dataclass alanlarını introspect et
        try:
            import dataclasses
            if dataclasses.is_dataclass(self.template):
                return dataclasses.asdict(self.template)
        except Exception:
            pass
        return None

    @staticmethod
    def _template_from_dict(d):
        if d is None:
            return None
        try:
            from pitch.template import PitchTemplate
            if hasattr(PitchTemplate, "from_dict"):
                return PitchTemplate.from_dict(d)
            # alanlardan yeniden kur
            return PitchTemplate(**{k: v for k, v in d.items()
                                    if k in getattr(PitchTemplate, "__dataclass_fields__", {})})
        except Exception:
            return None  # template.py yoksa: dims fallback ile devam edilir


# ------------------------------------------------------------------- CLI ----
def _main(argv):
    """Basit manuel-kalibrasyon CLI.

    Kullanım:
      python -m pitch.homography annotate <frame.png>
          -> noktaları tıkla, sonra her nokta için "X Y" (metre) gir, kaydet.
    """
    if len(argv) >= 2 and argv[0] == "annotate":
        frame_path = argv[1]
        camera_id = argv[2] if len(argv) > 2 else "cam"
        img_pts = PitchHomography.annotate_points(frame_path)
        if len(img_pts) < 4:
            print("en az 4 nokta gerek; iptal.", file=sys.stderr)
            return 1
        print(f"{len(img_pts)} nokta tıklandı. Her biri için saha metresi gir:")
        world = []
        for i, (px, py) in enumerate(img_pts):
            s = input(f"  nokta {i} (px={px:.0f},{py:.0f}) -> 'X_m Y_m': ")
            x, y = (float(t) for t in s.split())
            world.append((x, y))
        homo = PitchHomography(camera_id, None)
        qa = homo.calibrate_manual(img_pts, np.array(world))
        out = f"calib/{camera_id}.json"
        homo.save(out)
        print(f"kaydedildi -> {out}")
        print("QA:", json.dumps(qa, indent=2))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
