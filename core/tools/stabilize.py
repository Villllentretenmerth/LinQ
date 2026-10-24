"""Стабилизация ролика, снятого с рук: каждый кадр приводится к опорному (гомография по ORB, RANSAC).
Для поворота камеры на месте гомография точная; движущихся людей RANSAC отбрасывает как выбросы.
    python stabilize.py in.mp4 out.mp4 [--ref N]
"""
import argparse, json, subprocess
import cv2, numpy as np

SC = 0.75   # признаки ищем на 960x540

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--ref", type=int, default=0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orb = cv2.ORB_create(5000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    def feats(im):
        g = cv2.cvtColor(cv2.resize(im, None, fx=SC, fy=SC), cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        return orb.detectAndCompute(g, None)
    def match(fa, fb):
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None: return None, 0
        ms = [m for m, n in (p for p in bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.8 * n.distance]
        if len(ms) < 20: return None, 0
        A = np.float32([ka[m.queryIdx].pt for m in ms]) / SC
        B = np.float32([kb[m.trainIdx].pt for m in ms]) / SC
        H, inl = cv2.findHomography(A, B, cv2.USAC_MAGSAC, 2.5)
        return H, int(inl.sum()) if inl is not None else 0
    Hs = []; stats = []; ref = prev = None; prevH = np.eye(3)
    while True:
        ok, im = cap.read()
        if not ok: break
        f = feats(im)
        if ref is None:
            ref = prev = f; Hs.append(np.eye(3)); stats.append(9999); continue
        H, n = match(f, ref)                      # кадр -> опорный
        if H is None or n < 60:                   # слабо — через предыдущий кадр
            Hn, nn = match(f, prev)
            if Hn is not None:
                H, n = prevH @ Hn, -nn
        H = H if H is not None else prevH
        Hs.append(H); stats.append(n); prevH = H; prev = f
    cap.release(); cap = cv2.VideoCapture(a.src)
    # обратный проход для кадров до опорного не нужен при ref=0
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "18", "-pix_fmt", "yuv420p", a.dst], stdin=subprocess.PIPE)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    shifts = []
    for H in Hs:
        ok, im = cap.read()
        if not ok: break
        out = cv2.warpPerspective(im, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        proc.stdin.write(out.tobytes())
        shifts.append(float(np.abs(cv2.perspectiveTransform(corners, H) - corners).max()))
    proc.stdin.close(); proc.wait()
    weak = sum(1 for s in stats if 0 <= s < 60) + sum(1 for s in stats if s < 0)
    print(json.dumps({"file": a.src.split("/")[-1], "frames": len(Hs), "fps": round(fps, 3),
                      "min_inliers": int(min(abs(s) for s in stats)), "chained": sum(1 for s in stats if s < 0),
                      "max_correction_px": round(max(shifts), 1)}))
    np.save(a.dst.replace(".mp4", "_H.npy"), np.array(Hs))

if __name__ == "__main__":
    main()
