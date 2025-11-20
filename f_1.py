"""
Dobot Magician Lite — FINAL Maze Solver (Red → Green) + 4-Point Dobot Calibration (No AI)
----------------------------------------------------------------------------------------
✅ Detects 4×4 maze grid (robust to thick walls)
✅ Red = START, Green = GOAL (HSV robust indoors)
✅ Wall-aware adjacency for A*
✅ 4-point pixel→mm homography
✅ Sends path to Dobot (safe Z, draw Z), or dry-run if Dobot not connected
"""

import cv2 as cv
import numpy as np
import time
import os
import cv2 as cv
from dotenv import load_dotenv
import numpy as np
import time
import json
import requests

def some():
    load_dotenv()
    USE_AI_AGENT = True
    PERPLEXITY_MODEL = "sonar-pro"
    PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    if not PERPLEXITY_API_KEY:
        print("⚠️ PERPLEXITY_API_KEY not found — AI agent disabled.")
        USE_AI_AGENT = False
    return USE_AI_AGENT,PERPLEXITY_MODEL,PERPLEXITY_URL,PERPLEXITY_API_KEY

USE_AI_AGENT, PERPLEXITY_MODEL, PERPLEXITY_URL, PERPLEXITY_API_KEY = some()

# ====== Optional Dobot import (falls back to dry-run if not available) ======
ENABLE_DOBOT = True
try:
    import pydobot  # pip install pydobot
except Exception as e:
    print(f"⚠️ pydobot import failed ({e}). Running in dry-run mode.")
    ENABLE_DOBOT = False

# ===================== USER CONFIG =====================
CAMERA_ID =0
DOBOT_PORT = "COM15"        # ← set your Dobot port here (e.g., "COM13" on Windows, "/dev/ttyUSB0" on Linux)
SAFE_Z = 30.0               # Z above the surface for travel
DRAW_Z = -40.0              # Z for drawing/marking on the maze
PAUSE_BETWEEN_CMDS = 0.1    # seconds between linear moves

# --- Use absolute world millimeters (no offsets). If you want offsets, set these and flip the flag below.
USE_ABSOLUTE_WORLD_MM = True
OFFSET_X = 0.0
OFFSET_Y = 0.0

# --- 4-point calibration data (pixel → world mm). Order: TL, TR, BR, BL
#     Update these to match your setup. Pixel points are image coords, real-world are Dobot coords (mm).
REAL_WORLD_POINTS = np.array([
    [318.26,  52.36],   # Top-Left
    [315.06, -64.73],   # Top-Right
    [236.07, -64.25],   # Bottom-Right
    [242.54,  52.55],   # Bottom-Left
], dtype=np.float32)

PIXEL_POINTS = np.array([
    [121, 183],
    [387, 179],
    [397, 358],
    [137,354],
], dtype=np.float32)

# ===================== CAPTURE =====================
def capture_with_preview(cam_id=0, preview_time=10):
    cap = cv.VideoCapture(cam_id)
    if not cap.isOpened():
        raise IOError("Cannot open camera.")
    print(f"🎥 Live camera ON ({preview_time}s) – align maze correctly...")
    start = time.time()
    frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cv.imshow("Camera Preview", img)
        if time.time() - start >= preview_time:
            frame = img.copy()
            print("✅ Frame captured from camera.")
            break
        if cv.waitKey(1) & 0xFF == ord("q"):
            frame = img.copy()
            print("✅ Frame captured (manual).")
            break
    cap.release()
    cv.destroyAllWindows()
    if frame is None:
        raise RuntimeError("❌ No frame captured.")
    return frame
def auto_deskew(bgr, debug=False):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3,3), 0)
    edges = cv.Canny(gray, 60, 180)
    lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        print("⚠️ No Hough lines detected — skipping rotation correction.")
        return bgr.copy(), np.eye(2,3), np.eye(2,3), 0.0
    angles = [((np.degrees(l[0][1]) + 45) % 90) - 45 for l in lines]
    median_angle = np.median(angles)
    h,w = bgr.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)
    Minv = cv.invertAffineTransform(M)
    rotated = cv.warpAffine(bgr, M, (w,h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if debug:
        dbg = rotated.copy()
        cv.putText(dbg, f"Deskew {median_angle:.2f}°", (10,30), cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv.imshow("Deskew Debug", dbg); cv.waitKey(500)
    return rotated, M, Minv, median_angle
# ===================== MAZE GRID DETECTION =====================
def detect_maze_grid(bgr):
    """Detects the maze ROI and ensures a valid 4×4 grid. Returns rows, cols, v_lines, h_lines, bin_image."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(gray, 180, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)

    # Largest contour = maze outline
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("❌ Maze not found.")
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    roi = bgr[y:y + h, x:x + w]

    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    thr = cv.adaptiveThreshold(gray_roi, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY_INV, 19, 5)  # walls=white(255), gaps=black(0)

    # Auto thickness handling
    line_thickness = max(2, int(min(w, h) / 60))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, line_thickness * 6))
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (line_thickness * 6, 1))
    v_img = cv.morphologyEx(thr, cv.MORPH_OPEN, v_kernel)
    h_img = cv.morphologyEx(thr, cv.MORPH_OPEN, h_kernel)
    v_proj = v_img.sum(axis=0)
    h_proj = h_img.sum(axis=1)
    v_peaks = np.where(v_proj > 0.5 * v_proj.max())[0]
    h_peaks = np.where(h_proj > 0.5 * h_proj.max())[0]

    def group(p, tol=25):
        if len(p) == 0:
            return []
        groups, current = [], [p[0]]
        for val in p[1:]:
            if val - current[-1] <= tol:
                current.append(val)
            else:
                groups.append(current)
                current = [val]
        groups.append(current)
        return [int(np.mean(g)) for g in groups]

    v_lines_local = group(v_peaks)
    h_lines_local = group(h_peaks)

    v_lines = [x + int(i) for i in v_lines_local]
    h_lines = [y + int(i) for i in h_lines_local]

    # Guarantee at least 5x5 grid lines (4×4 cells)
    if len(v_lines) < 5:
        step = (v_lines[-1] - v_lines[0]) // 4
        v_lines = [v_lines[0] + step * i for i in range(5)]
    if len(h_lines) < 5:
        step = (h_lines[-1] - h_lines[0]) // 4
        h_lines = [h_lines[0] + step * i for i in range(5)]

    # Debug visualization
    debug = bgr.copy()
    for xv in v_lines:
        cv.line(debug, (xv, y), (xv, y + h), (255, 0, 0), 2)
    for yh in h_lines:
        cv.line(debug, (x, yh), (x + w, yh), (0, 255, 0), 2)
    cv.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv.imshow("Grid Debug", debug)
    print("🧩 Showing grid debug (2 s)...")
    cv.waitKey(2000)
    cv.destroyWindow("Grid Debug")

    # Return the thresholded (not inverted) image as "bin_img" for wall-aware checks
    bin_img = thr
    return len(h_lines) - 1, len(v_lines) - 1, v_lines, h_lines, bin_img
def perplexity_astar_agent(start,goal,adj):
    print("\n🧠 [Agentic AI] Perplexity-based A* path planning...")
    if not USE_AI_AGENT or not PERPLEXITY_API_KEY:
        
        return astar(start,goal,adj)
    adj_dict={f"{r},{c}":[f"{rr},{cc}" for rr,cc in nbs] for (r,c),nbs in adj.items()}
    prompt=(f"You are an A* path planning assistant. "
            f"Given adjacency of open cells, start and goal, "
            f"return JSON {{\"path\": [[r,c],...]}}.\n"
            f"start={start}, goal={goal}, adjacency={json.dumps(adj_dict)}")
    payload={"model":PERPLEXITY_MODEL,
             "messages":[{"role":"system","content":"Output only JSON."},
                         {"role":"user","content":prompt}],
             "max_tokens":200,"temperature":0.1}
    headers={"Authorization":f"Bearer {PERPLEXITY_API_KEY}","Content-Type":"application/json"}
    try:
        r=requests.post(PERPLEXITY_URL,headers=headers,json=payload,timeout=40)
        if r.status_code!=200:
            
            return astar(start,goal,adj)
        msg=r.json().get("choices",[{}])[0].get("message",{}).get("content","")
        j=json.loads(msg)
        path=j.get("path",[])
        if isinstance(path,list) and all(len(p)==2 for p in path):
            print(f"✅ Path received ({len(path)} steps).")
            return [tuple(p) for p in path]
        print("⚠️ Invalid response — using local A*.")
        return astar(start,goal,adj)
    except Exception as e:
        print(f"⚠️ Agent error: {e}")
        return astar(start,goal,adj)

# ===================== COLOR DETECTION =====================
def detect_red_and_green_in_roi(frame, v_lines, h_lines):
    if not v_lines or not h_lines:
        return None, None
    x0, x1 = v_lines[0], v_lines[-1]
    y0, y1 = h_lines[0], h_lines[-1]
    roi = frame[y0:y1, x0:x1]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # Robust ranges for indoor lighting
    red_mask = (cv.inRange(hsv, (0, 120, 50), (10, 255, 255)) |
                cv.inRange(hsv, (170, 120, 50), (180, 255, 255)))
    green_mask = (cv.inRange(hsv, (25, 40, 50), (85, 255, 255)) |
                  cv.inRange(hsv, (25, 40, 20), (85, 255, 150)))

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)

    def center_from_largest(mask):
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv.contourArea)
        M = cv.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + x0
        cy = int(M["m01"] / M["m00"]) + y0
        return (cx, cy)

    return center_from_largest(red_mask), center_from_largest(green_mask)

# ===================== WALL-AWARE ADJACENCY =====================
def build_adjacency(bin_img, v_lines, h_lines, wall_check_width=10, open_threshold=80):
    """
    Builds adjacency between cells using wall-aware corridor detection.
    Uses the non-inverted binary image: walls=255, openings=0.
    If the mean in the strip is LOW (near 0), we consider it OPEN.
    """
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1
    adj = {(i, j): [] for i in range(rows) for j in range(cols)}

    # Horizontal neighbors (left↔right)
    for i in range(rows):
        for j in range(cols - 1):
            xw = v_lines[j + 1]
            y0, y1 = h_lines[i] + 2, h_lines[i + 1] - 2
            strip = bin_img[y0:y1, xw - wall_check_width:xw + wall_check_width]
            if strip.size > 0 and np.mean(strip) < open_threshold:
                adj[(i, j)].append((i, j + 1))
                adj[(i, j + 1)].append((i, j))

    # Vertical neighbors (up↔down)
    for i in range(rows - 1):
        for j in range(cols):
            yw = h_lines[i + 1]
            x0, x1 = v_lines[j] + 2, v_lines[j + 1] - 2
            strip = bin_img[yw - wall_check_width:yw + wall_check_width, x0:x1]
            if strip.size > 0 and np.mean(strip) < open_threshold:
                adj[(i, j)].append((i + 1, j))
                adj[(i + 1, j)].append((i, j))

    # Optional Debug: visualize adjacency checks
    debug_adj = cv.cvtColor(bin_img, cv.COLOR_GRAY2BGR)
    for i in range(rows):
        for j in range(cols - 1):
            xw = v_lines[j + 1]
            y0, y1 = h_lines[i] + 2, h_lines[i + 1] - 2
            cv.line(debug_adj, (xw, y0), (xw, y1), (0, 255, 255), 1)
    for i in range(rows - 1):
        for j in range(cols):
            yw = h_lines[i + 1]
            x0, x1 = v_lines[j] + 2, v_lines[j + 1] - 2
            cv.line(debug_adj, (x0, yw), (x1, yw), (255, 255, 0), 1)
    cv.imshow("Adjacency Check", debug_adj)
    cv.waitKey(500)
    cv.destroyWindow("Adjacency Check")

    return adj

# ===================== A* =====================
def astar(start, goal, adj):
    def h(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
    if None in start or None in goal:
        return []
    open_set = {start}
    came = {}
    g = {start: 0}
    f = {start: h(start, goal)}
    while open_set:
        cur = min(open_set, key=lambda x: f.get(x, 1e9))
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        open_set.remove(cur)
        for nb in adj.get(cur, []):
            t = g[cur] + 1
            if t < g.get(nb, 1e9):
                came[nb] = cur
                g[nb] = t
                f[nb] = t + h(nb, goal)
                open_set.add(nb)
    return []

# ===================== HOMOGRAPHY & TRANSFORM =====================
def compute_homography_manual():
    H, _ = cv.findHomography(PIXEL_POINTS, REAL_WORLD_POINTS)
    if H is None:
        raise RuntimeError("❌ Homography computation failed. Check your 4 points.")
    return H

def pixels_to_mm(path_pixels, H, offset_x=0.0, offset_y=0.0, maze_height_mm=None):
    """
    Transforms a list of (x,y) pixel points into (X,Y) mm using homography.
    If USE_ABSOLUTE_WORLD_MM is False and maze_height_mm is provided, flip Y and add offsets.
    """
    if len(path_pixels) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.array(path_pixels, np.float32).reshape(-1, 1, 2)
    pts_mm = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    if USE_ABSOLUTE_WORLD_MM:
        return pts_mm
    # Optional alternative frame handling
    if maze_height_mm is None:
        maze_height_mm = 170.0
    pts_mm[:, 1] = maze_height_mm - pts_mm[:, 1]
    pts_mm[:, 0] += offset_x
    pts_mm[:, 1] += offset_y
    return pts_mm

# ===================== DOBOT =====================
class DobotController:
    def __init__(self, port):
        self.port = port
        self.device = None

    def connect(self):
        if not ENABLE_DOBOT:
            print("🔒 Dobot disabled (dry-run).")
            return
        try:
            self.device = pydobot.Dobot(port=self.port)
            print(f"✅ Connected to Dobot ({self.port})")
        except Exception as e:
            print(f"⚠️ Dobot connection failed: {e}")
            self.device = None

    def move_linear(self, x, y, z, r=0.0, wait=False):
        if self.device is None:
            print(f"[DRY-RUN] Move to (X={x:.1f}, Y={y:.1f}, Z={z:.1f})")
            time.sleep(PAUSE_BETWEEN_CMDS)
            return
        self.device.move_to(x, y, z, r, wait=wait)
        time.sleep(PAUSE_BETWEEN_CMDS)

    def disconnect(self):
        if self.device:
            self.device.close()
            print("🔌 Dobot disconnected.")

# ===================== MAIN =====================
def main():
    # 1) Capture and detect grid
    frame = capture_with_preview(CAMERA_ID, preview_time=10)
    frame_rot,M,Minv,ang=auto_deskew(frame,debug=True)
    rows, cols, v_lines, h_lines, bin_img = detect_maze_grid(frame_rot)
    print(f"Grid detected: {rows}x{cols}")

    vis = frame_rot.copy()
    if v_lines and h_lines:
        cv.rectangle(vis, (v_lines[0], h_lines[0]), (v_lines[-1], h_lines[-1]), (0, 255, 0), 2)
        for x in v_lines: cv.line(vis, (x, h_lines[0]), (x, h_lines[-1]), (255, 255, 255), 1)
        for y in h_lines: cv.line(vis, (v_lines[0], y), (v_lines[-1], y), (255, 255, 255), 1)
        print("🟩 Maze ROI detected.")
    else:
        print("❌ Maze grid detection failed.")
    if np.mean(bin_img) < 127:
        print("🌀 Maze polarity: black walls, white paths.")
    else:
        print("🌀 Maze polarity: white walls, black paths. Inverting.")
        bin_img = cv.bitwise_not(bin_img)
    # 2) Detect start (red) and goal (green)
    red_pt, green_pt = detect_red_and_green_in_roi(frame_rot,v_lines,h_lines)
    if red_pt:
        cv.circle(vis, red_pt, 10, (0, 0, 255), -1)
        cv.putText(vis, "RED", (red_pt[0] + 10, red_pt[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if green_pt:
        cv.circle(vis, green_pt, 10, (0, 255, 0), -1)
        cv.putText(vis, "GREEN", (green_pt[0] + 10, green_pt[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def pixel_to_cell(pt):
        if pt is None: return (None, None)
        x, y = pt
        c = next((j for j in range(len(v_lines) - 1) if v_lines[j] <= x < v_lines[j + 1]), None)
        r = next((i for i in range(len(h_lines) - 1) if h_lines[i] <= y < h_lines[i + 1]), None)
        return (r, c)
    
    red_cell = pixel_to_cell(red_pt)
    green_cell = pixel_to_cell(green_pt)
    print(f"RED cell: {red_cell}, GREEN cell: {green_cell}")

    # 3) Build adjacency & A*
    start_choice=input("Which color is START? (r/g): ").strip().lower()
    start_cell,goal_cell=(red_cell,green_cell) if start_choice=="r" else (green_cell,red_cell)
    adj = build_adjacency(bin_img, v_lines, h_lines, wall_check_width=7, open_threshold=80)
    path_cells = perplexity_astar_agent(start_cell,goal_cell,adj) \
                 if USE_AI_AGENT else astar(start_cell,goal_cell,adj)

    # 4) Visualize path in image
    path_pixels = []
    if path_cells:
        def cell_center(rc):
            r, c = rc
            x = int((v_lines[c] + v_lines[c + 1]) / 2)
            y = int((h_lines[r] + h_lines[r + 1]) / 2)
            return (x, y)
        for i in range(len(path_cells) - 1):
            p1, p2 = cell_center(path_cells[i]), cell_center(path_cells[i + 1])
            path_pixels.append(p1) if i == 0 else None
            path_pixels.append(p2)
            cv.line(vis, p1, p2, (0, 255, 255), 3)
        print("✅ Correct wall-safe path drawn.")
    else:
        print("⚠️ Path not found (try tuning wall_check_width or open_threshold).")

    cv.imshow("Maze Detection + Dots + Path", vis)
    print("👀 Showing final result for 5 seconds...")
    cv.waitKey(5000)
    cv.destroyAllWindows()

    if not path_pixels:
        print("❌ No path pixels to send to Dobot.")
        return

    # 5) Homography & pixel→mm conversion
    H = compute_homography_manual()
    pts_mm = pixels_to_mm(path_pixels, H, OFFSET_X, OFFSET_Y, maze_height_mm=170.0)

    print("\n🧭 Waypoints (Dobot coordinates in mm):")
    for i, (x, y) in enumerate(pts_mm):
        print(f"{i:02d}: X={x:.2f}  Y={y:.2f}  Z={DRAW_Z:.2f}")
    print(f"End lift: X={pts_mm[-1][0]:.2f}  Y={pts_mm[-1][1]:.2f}  Z={SAFE_Z:.2f}")

    # 6) Send moves to Dobot (or dry-run)
    bot = DobotController(DOBOT_PORT)
    bot.connect()

    # Move above start, descend to draw Z, follow path, then lift
    sx, sy = pts_mm[0]
    bot.move_linear(sx, sy, SAFE_Z)
    bot.move_linear(sx, sy, DRAW_Z)
    for x, y in pts_mm[1:]:
        bot.move_linear(x, y, DRAW_Z)
    ex, ey = pts_mm[-1]
    bot.move_linear(ex, ey, SAFE_Z)
    bot.disconnect()

    print("✅ Done.")

if __name__ == "__main__":
    main()
