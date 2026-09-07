import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
import queue
from PIL import Image, ImageTk
import sys
import os
from urllib.request import urlretrieve
from screeninfo import get_monitors

import pygame # For audio alerts

try:
    import mediapipe as mp
except ImportError:
    mp = None

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
DRINK_OBJECT_LABELS = {"bottle", "cup", "wine glass"}
HELD_DRINK_MIN_SCORE = 0.18
FACE_ONLY_DRINK_MIN_SCORE = 0.42
MAX_DRINK_AREA_RATIO = 0.28
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 13, 14, 0]
LOWER_FACE_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 199, 200, 152, 148, 176, 400, 377]
FINGER_TIP_LANDMARKS = {4, 8, 12, 16, 20}
FINGER_EDGE_LANDMARKS = {3, 7, 11, 15, 19}
PALM_LANDMARKS = {0, 1, 2, 5, 9, 13, 17}
HAND_CONTACT_WEIGHTS = {
    4: 1.7, 8: 1.7, 12: 1.55, 16: 1.45, 20: 1.35,
    3: 1.25, 7: 1.25, 11: 1.15, 15: 1.10, 19: 1.05,
    2: 0.95, 5: 0.95, 6: 0.95, 9: 0.90, 10: 0.90,
    13: 0.85, 14: 0.85, 17: 0.80, 18: 0.80, 0: 0.65, 1: 0.70,
}

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class BeardTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Beard Picking Tracker")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.colors = {
            "bg": "#eef2f6",
            "panel": "#ffffff",
            "panel_alt": "#f7f9fc",
            "border": "#d9e2ec",
            "text": "#172033",
            "muted": "#64748b",
            "green": "#16a34a",
            "green_dark": "#15803d",
            "red": "#dc2626",
            "red_dark": "#b91c1c",
            "blue": "#2563eb",
            "dark": "#0f172a",
        }
        self.root.configure(bg=self.colors["bg"])
        
        # State
        self.is_tracking = False
        self.is_alert_active = False
        self.alert_count = 0
        self.cap = None
        self.hands = None
        self.face_mesh = None
        self.object_detector = None
        self.object_detector_loading = False
        self.alert_end_time = 0
        self.flash_state = False # For flashing text
        self.flash_job = None # To cancel flashing
        self.is_muted = False # Mute state
        self.audio_start_job = None # To schedule audio start
        self.ui_queue = queue.Queue()
        
        # Audio Initialization

        self.alert_sound = None
        try:
            pygame.mixer.init()
            mp3_path = resource_path("alert.mp3")
            wav_path = resource_path("alert.wav")
            
            if os.path.exists(mp3_path):
                self.alert_sound = pygame.mixer.Sound(mp3_path)
            elif os.path.exists(wav_path):
                self.alert_sound = pygame.mixer.Sound(wav_path)
            else:
                print("WARNING: alert.mp3/wav not found.")
        except Exception as e:
            print(f"Audio init failed: {e}")
        
        # Performance/Optimization State
        self.last_detection = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.last_object_results = []
        self.last_object_timestamp_ms = -1
        self.object_frame_count = 0
        self.last_drink_seen_time = 0
        self.last_sip_guard_time = 0
        self.primary_face_rect = None

        # Detection state
        self.touch_streak = 0           # consecutive inference frames with touch
        self.contact_score = 0.0
        self.in_alert_state = False     # latched alert state (with hysteresis)

        # Settings
        self.sensitivity = tk.DoubleVar(value=0.18) # Face-zone padding. Higher = easier to trigger.
        self.alert_duration = tk.DoubleVar(value=0.1) # Default duration 0.1s
        self.audio_delay = tk.DoubleVar(value=0.6) # Default delay 0.6s

        self.frame_skip = tk.IntVar(value=6) # Process every Nth preview frame
        self.preview_delay_ms = 100 # 10 FPS preview cap for low CPU use.

        # Strict-FP settings
        self.required_streak = tk.IntVar(value=1)         # consecutive inference frames before alert
        self.conf_threshold = tk.DoubleVar(value=0.0)     # optional handedness confidence floor
        self.capture_resolution = tk.StringVar(value="480x360")
        self.drink_object_guard = tk.BooleanVar(value=True)
        self.drink_object_interval = tk.IntVar(value=8)
        self.drink_memory_seconds = tk.DoubleVar(value=4.0)
        self.seltzer_can_guard = tk.BooleanVar(value=True)
        self.sip_guard_memory_seconds = tk.DoubleVar(value=2.5)


        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Tracker.TCombobox", fieldbackground=self.colors["panel"], background=self.colors["panel"])

        app_frame = tk.Frame(root, bg=self.colors["bg"])
        app_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        sidebar = tk.Frame(app_frame, bg=self.colors["panel"], width=380, highlightbackground=self.colors["border"], highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        content = tk.Frame(app_frame, bg=self.colors["bg"])
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 0))

        tk.Label(sidebar, text="No Picking", font=("Segoe UI", 26, "bold"), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w", padx=22, pady=(22, 0))
        tk.Label(sidebar, text="Hand-aware beard tracker", font=("Segoe UI", 11), bg=self.colors["panel"], fg=self.colors["muted"]).pack(anchor="w", padx=24, pady=(2, 18))

        status_frame = tk.Frame(sidebar, bg=self.colors["panel_alt"], highlightbackground=self.colors["border"], highlightthickness=1)
        status_frame.pack(fill=tk.X, padx=18, pady=(0, 18))
        tk.Label(status_frame, text="Status", font=("Segoe UI", 9, "bold"), bg=self.colors["panel_alt"], fg=self.colors["muted"]).pack(anchor="w", padx=14, pady=(12, 0))
        self.status_label = tk.Label(status_frame, text="Idle - detectors loading", font=("Segoe UI", 12, "bold"), bg=self.colors["panel_alt"], fg=self.colors["text"], wraplength=300, justify=tk.LEFT)
        self.status_label.pack(anchor="w", padx=14, pady=(2, 10))
        self.count_label = tk.Label(status_frame, text="Alerts: 0", font=("Segoe UI", 12, "bold"), bg=self.colors["panel_alt"], fg=self.colors["blue"])
        self.count_label.pack(anchor="w", padx=14, pady=(0, 12))

        btn_frame = tk.Frame(sidebar, bg=self.colors["panel"])
        btn_frame.pack(fill=tk.X, padx=18, pady=(0, 18))

        self.start_btn = self._make_button(btn_frame, "Start", self.start_tracking, self.colors["green"], self.colors["green_dark"])
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.stop_btn = self._make_button(btn_frame, "Stop", self.stop_tracking, self.colors["red"], self.colors["red_dark"], state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.mute_btn = self._make_button(btn_frame, "Mute", self.toggle_mute, "#475569", "#334155")
        self.mute_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        settings_shell = tk.Frame(sidebar, bg=self.colors["panel"])
        settings_shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=(2, 18))
        settings_canvas = tk.Canvas(settings_shell, bg=self.colors["panel"], highlightthickness=0, bd=0)
        settings_scrollbar = ttk.Scrollbar(settings_shell, orient=tk.VERTICAL, command=settings_canvas.yview)
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        settings_frame = tk.Frame(settings_canvas, bg=self.colors["panel"])
        settings_window = settings_canvas.create_window((0, 0), window=settings_frame, anchor="nw")
        settings_frame.bind("<Configure>", lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        settings_canvas.bind("<Configure>", lambda e: settings_canvas.itemconfigure(settings_window, width=e.width))
        settings_canvas.bind("<Enter>", lambda e: self._bind_mousewheel(settings_canvas))
        settings_canvas.bind("<Leave>", lambda e: self._unbind_mousewheel())

        tk.Label(settings_frame, text="Settings", font=("Segoe UI", 13, "bold"), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w", pady=(0, 10))

        self.slider = self._make_scale(settings_frame, "Sensitivity", self.sensitivity, 0.0, 0.35, 0.01)
        self.duration_slider = self._make_scale(settings_frame, "Alert Duration", self.alert_duration, 0.1, 5.0, 0.1)
        self.skip_slider = self._make_scale(settings_frame, "CPU Saver", self.frame_skip, 1, 6, 1)
        self.delay_slider = self._make_scale(settings_frame, "Audio Delay", self.audio_delay, 0.0, 5.0, 0.1)
        self.streak_slider = self._make_scale(settings_frame, "Required Streak", self.required_streak, 1, 6, 1)
        self.conf_slider = self._make_scale(settings_frame, "Hand Confidence Filter", self.conf_threshold, 0.0, 0.95, 0.05)
        self._make_check(settings_frame, "Drink Object Detector", self.drink_object_guard)
        self.drink_object_guard.trace_add("write", lambda *_: self.load_object_detector_thread())
        self.object_interval_slider = self._make_scale(settings_frame, "Drink Scan Interval", self.drink_object_interval, 2, 12, 1)
        self.drink_memory_slider = self._make_scale(settings_frame, "Drink Memory", self.drink_memory_seconds, 0.5, 6.0, 0.5)
        self._make_check(settings_frame, "Seltzer Can Sip Guard", self.seltzer_can_guard)
        self.sip_memory_slider = self._make_scale(settings_frame, "Sip Guard Memory", self.sip_guard_memory_seconds, 0.3, 3.0, 0.1)

        res_frame = tk.Frame(settings_frame, bg=self.colors["panel"])
        res_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(res_frame, text="Capture Resolution", font=("Segoe UI", 10, "bold"), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w")
        self.res_menu = ttk.Combobox(res_frame, textvariable=self.capture_resolution,
                                     values=["480x360", "640x480", "960x720", "Native"],
                                     state="readonly", width=12, style="Tracker.TCombobox")
        self.res_menu.pack(anchor="w", pady=(5, 0))

        header = tk.Frame(content, bg=self.colors["bg"])
        header.pack(fill=tk.X, pady=(0, 12))
        tk.Label(header, text="Live Detection", font=("Segoe UI", 20, "bold"), bg=self.colors["bg"], fg=self.colors["text"]).pack(side=tk.LEFT)
        tk.Label(header, text="480x360 default - 10 FPS preview", font=("Segoe UI", 10), bg=self.colors["bg"], fg=self.colors["muted"]).pack(side=tk.RIGHT, pady=(8, 0))

        video_frame = tk.Frame(content, bg=self.colors["dark"], highlightbackground=self.colors["border"], highlightthickness=1)
        video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(video_frame, bg=self.colors["dark"])
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Alert Windows (One per monitor)
        self.alert_windows = []
        self.alert_labels = [] # To reference text for flashing (Now stores canvas, item_id tuples)

        self.vignette_images = [] # Keep references to prevent GC
        self.create_alert_windows()

        # Load Model
        self.root.after(100, self.process_ui_queue)
        self.load_model_thread()

    def generate_vignette(self, width, height):
        # Create a radial gradient from center
        # Center = Black (0,0,0), Edge = Red (255,0,0)
        
        # Coordinates
        Y = np.linspace(-1, 1, height)
        X = np.linspace(-1, 1, width)
        X, Y = np.meshgrid(X, Y)
        
        # Radius
        R = np.sqrt(X**2 + Y**2)
        
        # Vignette mask: 0 at center, 1 at edges. 
        # Use power to make the center "safe zone" larger and edges steeper
        # R = 0 -> Mask = 0
        # R > 0.5 -> Mask increases
        Mask = np.clip(R, 0, 1) 
        Mask = Mask ** 2 # Lowered from 3 for softer transition
        
        # Create image (H, W, 3)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Red channel scales with Mask
        img[:, :, 0] = (Mask * 255).astype(np.uint8)
        # Green/Blue stay 0 (Black base)
        
        return Image.fromarray(img)

    def create_alert_windows(self):
        # Clean up existing if any
        for win in self.alert_windows:
            win.destroy()
        self.alert_windows = []
        self.alert_labels = []
        self.vignette_images = []

        try:
            monitors = get_monitors()
            # If no monitors detected (e.g. RDP/VM sometimes), fall back
            if not monitors: raise Exception("No monitors found")
            
            for m in monitors:
                self._create_single_window(m.width, m.height, m.x, m.y)
                
        except Exception as e:
            print(f"Error checking monitors: {e}. Fallback to fullscreen.")
            self._create_single_window(self.root.winfo_screenwidth(), self.root.winfo_screenheight(), 0, 0)

    def _create_single_window(self, w, h, x, y):
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        
        # Vignette & Transparency
        win.configure(bg='black')
        try:
             # REMOVED: win.attributes("-transparentcolor", "black") - Causes holes in the window
             win.attributes("-alpha", 0.75) # Semi-transparent blend
        except:

            print("Transparency not supported on this platform/config")

        # Create Canvas for transparent text rendering
        # bg='black' ensures the background maps to the transparent key
        canvas = tk.Canvas(win, bg='black', highlightthickness=0)
        canvas.pack(fill='both', expand=True)

        # Generate Background
        pil_img = self.generate_vignette(w, h)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.vignette_images.append(tk_img) # Keep ref
        
        # Draw Vignette Image
        canvas.create_image(0, 0, image=tk_img, anchor="nw")

        # Text Setup
        cx, cy = w / 2, h / 2
        text_content = "STOP PICKING"
        font_spec = ("Arial", 150, "bold")
        
        # Outline (Thick Black)
        # We use #020202 to ensure it is NOT treated as transparent (if black is key)
        outline_color = "#020202"
        offsets = [
            (-3, -3), (-3, 0), (-3, 3),
            (0, -3),           (0, 3),
            (3, -3),  (3, 0),  (3, 3)
        ]
        
        for ox, oy in offsets:
            canvas.create_text(
                cx + ox, cy + oy,
                text=text_content,
                font=font_spec,
                fill=outline_color,
                anchor="center"
            )

        # Main Text (Target for flashing)
        label_id = canvas.create_text(
            cx, cy,
            text=text_content,
            font=font_spec,
            fill="red",
            anchor="center"
        )
        self.alert_labels.append((canvas, label_id))
        
        # FORCE QUIT BUTTON (Top Right)
        # Placed on top of canvas using win.place (Button is child of win, so it floats)
        quit_btn = self._make_button(
            win,
            "FORCE QUIT",
            self.force_quit,
            self.colors["red"],
            self.colors["red_dark"],
        )
        quit_btn.place(relx=0.98, rely=0.03, anchor="ne")
        quit_btn.lift()

        win.bind("<Escape>", lambda e: self.hide_alert())
        self.alert_windows.append(win)

    def _make_button(self, parent, text, command, bg, active_bg, state=tk.NORMAL):
        return tk.Button(
            parent,
            text=text,
            command=command,
            state=state,
            font=("Segoe UI", 11, "bold"),
            bg=bg,
            activebackground=active_bg,
            fg="white",
            activeforeground="white",
            disabledforeground="#cbd5e1",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            cursor="hand2",
        )

    def _make_scale(self, parent, label, variable, from_, to, resolution):
        row = tk.Frame(parent, bg=self.colors["panel"])
        row.pack(fill=tk.X, pady=(0, 9))
        tk.Label(row, text=label, font=("Segoe UI", 10, "bold"), bg=self.colors["panel"], fg=self.colors["text"]).pack(anchor="w")
        scale = tk.Scale(
            row,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            activebackground=self.colors["blue"],
            troughcolor="#e2e8f0",
            highlightthickness=0,
            bd=0,
            length=300,
            font=("Segoe UI", 9),
        )
        scale.pack(fill=tk.X)
        return scale

    def _make_check(self, parent, label, variable):
        check = tk.Checkbutton(
            parent,
            text=label,
            variable=variable,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            activebackground=self.colors["panel"],
            activeforeground=self.colors["text"],
            selectcolor=self.colors["panel_alt"],
            anchor="w",
            padx=0,
            pady=6,
        )
        check.pack(fill=tk.X, pady=(0, 6))
        return check

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", lambda e: widget.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _unbind_mousewheel(self):
        self.root.unbind_all("<MouseWheel>")

    def force_quit(self):
        if self.cap:
            self.cap.release()
        self.close_detectors()
        self.root.destroy()
        sys.exit(0)

    def close_detectors(self):
        for detector_name in ("hands", "face_mesh", "object_detector"):
            detector = getattr(self, detector_name, None)
            if detector:
                try:
                    detector.close()
                except Exception:
                    pass
                setattr(self, detector_name, None)
        self.object_detector_loading = False

    def set_status_async(self, text):
        self.ui_queue.put(("status", text))

    def process_ui_queue(self):
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_label.config(text=value)
        except queue.Empty:
            pass

        try:
            if self.root.winfo_exists():
                self.root.after(100, self.process_ui_queue)
        except tk.TclError:
            pass

    def ensure_task_model(self, filename, url, label):
        model_dir = resource_path("models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            self.set_status_async(f"Downloading {label}")
            urlretrieve(url, model_path)
        return model_path

    def load_model_thread(self):
        def _load():
            try:
                if mp is None:
                    raise ImportError("mediapipe is not installed. Run: pip install -r requirements.txt")

                from mediapipe.tasks.python import vision

                hand_model = self.ensure_task_model("hand_landmarker.task", HAND_MODEL_URL, "hand detector")
                face_model = self.ensure_task_model("face_landmarker.task", FACE_MODEL_URL, "face detector")
                running_mode = vision.RunningMode.VIDEO

                hand_options = vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
                    running_mode=running_mode,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                face_options = vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=face_model),
                    running_mode=running_mode,
                    num_faces=3,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.hands = vision.HandLandmarker.create_from_options(hand_options)
                self.face_mesh = vision.FaceLandmarker.create_from_options(face_options)
                self.set_status_async("Ready")
                if self.drink_object_guard.get():
                    self.load_object_detector_thread()
            except Exception as e:
                error = str(e)
                self.set_status_async(f"Detector error: {error}")
        
        Thread(target=_load, daemon=True).start()

    def load_object_detector_thread(self):
        if not self.drink_object_guard.get():
            return
        if self.object_detector is not None or self.object_detector_loading:
            return

        self.object_detector_loading = True

        def _load():
            try:
                if mp is None:
                    raise ImportError("mediapipe is not installed")

                from mediapipe.tasks.python import vision

                object_model = self.ensure_task_model("efficientdet_lite0.tflite", OBJECT_MODEL_URL, "drink object detector")
                object_options = vision.ObjectDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=object_model),
                    running_mode=vision.RunningMode.VIDEO,
                    max_results=8,
                    score_threshold=0.18,
                    category_allowlist=list(DRINK_OBJECT_LABELS),
                )
                self.object_detector = vision.ObjectDetector.create_from_options(object_options)
                self.set_status_async("Drink detector ready")
            except Exception as e:
                self.object_detector = None
                self.set_status_async(f"Drink detector unavailable: {e}")
            finally:
                self.object_detector_loading = False

        self.set_status_async("Loading drink detector")
        Thread(target=_load, daemon=True).start()

    def start_tracking(self):
        if self.is_tracking:
            return
        if self.hands is None or self.face_mesh is None:
            self.status_label.config(text="Detectors still loading")
            return
        self.is_tracking = True
        self.alert_count = 0
        self.count_label.config(text=f"Alerts: {self.alert_count}")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Tracking")
        
        # Reset detection state
        self.touch_streak = 0
        self.contact_score = 0.0
        self.in_alert_state = False
        self.last_detection = None
        self.frame_count = 0
        self.last_object_results = []
        self.last_object_timestamp_ms = -1
        self.object_frame_count = 0
        self.last_drink_seen_time = 0
        self.last_sip_guard_time = 0
        self.primary_face_rect = None

        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video source")
            self.cap.set(cv2.CAP_PROP_FPS, 10)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Keep camera input small. MediaPipe is accurate at this size and it
            # avoids burning CPU on pixels we do not need.
            res = self.capture_resolution.get()
            if res != "Native":
                try:
                    w_str, h_str = res.split("x")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w_str))
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h_str))
                except Exception as e:
                    print(f"Could not set capture resolution {res}: {e}")

            self.process_video()
        except Exception as e:
            print(f"Error starting tracking: {e}")
            self.stop_tracking()



    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.mute_btn.config(text="Unmute", bg="#d97706", activebackground="#b45309")
            if self.alert_sound: self.alert_sound.stop()
        else:
            self.mute_btn.config(text="Mute", bg="#475569", activebackground="#334155")
            # If alert is currently active, restart sound
            if self.is_alert_active and self.alert_sound:
                self.alert_sound.play(loops=-1)

    def stop_tracking(self):
        self.is_tracking = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")

        # Reset detection state so a fresh Start doesn't carry stale streak/alert
        self.touch_streak = 0
        self.contact_score = 0.0
        self.in_alert_state = False
        self.last_detection = None
        self.last_sip_guard_time = 0

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image='')
        self.hide_alert()

    def show_alert(self):
        if not self.is_alert_active:
            self.is_alert_active = True
            self.alert_count += 1
            self.count_label.config(text=f"Alerts: {self.alert_count}")
            for win in self.alert_windows:
                win.deiconify()
                win.attributes("-topmost", True) # Reinforce on top
            
            # Start flashing
            self.flash_state = True
            self.flash_alert_loop()
            
            # TRIGGER AUDIO ALERT (With Delay)
            delay_ms = int(self.audio_delay.get() * 1000)
            if delay_ms <= 0:
                self._play_audio_now()
            else:
                self.audio_start_job = self.root.after(delay_ms, self._play_audio_now)

    def _play_audio_now(self):
        if self.is_alert_active and self.alert_sound and not self.is_muted:
            self.alert_sound.play(loops=-1)

    def hide_alert(self):
        if self.is_alert_active:
            self.is_alert_active = False
            for win in self.alert_windows:
                win.withdraw()
            
            # Stop flashing
            if self.flash_job:
                self.root.after_cancel(self.flash_job)
                self.flash_job = None
            # Reset color
            for canvas, label_id in self.alert_labels:
                canvas.itemconfigure(label_id, fill="red")
            
            # Cancel pending audio start if it hasn't happened yet
            if self.audio_start_job:
                self.root.after_cancel(self.audio_start_job)
                self.audio_start_job = None

            # Stop Audio
            if self.alert_sound:
                self.alert_sound.stop()


    def flash_alert_loop(self):
        if not self.is_alert_active: return
        
        # Toggle Color
        # Flashing Red <-> Black
        # Black text on Black background = Invisible
        # This creates a blinking effect.
        new_color = "black" if self.flash_state else "red"
        
        for canvas, label_id in self.alert_labels:
            canvas.itemconfigure(label_id, fill=new_color)

            
        self.flash_state = not self.flash_state
        # Schedule next flash (200ms)
        self.flash_job = self.root.after(200, self.flash_alert_loop)

    def _point_in_rect(self, point, rect):
        x, y = point
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def _drink_memory_active(self):
        memory = max(0.0, self.drink_memory_seconds.get())
        return (time.time() - self.last_drink_seen_time) <= memory

    def _sip_guard_memory_active(self):
        if not self.seltzer_can_guard.get():
            return False
        memory = max(0.0, self.sip_guard_memory_seconds.get())
        return (time.time() - self.last_sip_guard_time) <= memory

    def _rects_intersect(self, a, b):
        a_left, a_top, a_right, a_bottom = a
        b_left, b_top, b_right, b_bottom = b
        return not (a_right < b_left or b_right < a_left or a_bottom < b_top or b_bottom < a_top)

    def _rect_area(self, rect):
        left, top, right, bottom = rect
        return max(0, right - left) * max(0, bottom - top)

    def _rect_iou(self, a, b):
        a_left, a_top, a_right, a_bottom = a
        b_left, b_top, b_right, b_bottom = b
        inter_left = max(a_left, b_left)
        inter_top = max(a_top, b_top)
        inter_right = min(a_right, b_right)
        inter_bottom = min(a_bottom, b_bottom)
        inter_area = self._rect_area((inter_left, inter_top, inter_right, inter_bottom))
        union_area = self._rect_area(a) + self._rect_area(b) - inter_area
        if union_area <= 0:
            return 0
        return inter_area / union_area

    def _rect_intersection(self, a, b):
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _expand_rect(self, rect, pad_x, pad_y, width, height):
        left, top, right, bottom = rect
        return (
            int(max(0, left - pad_x)),
            int(max(0, top - pad_y)),
            int(min(width - 1, right + pad_x)),
            int(min(height - 1, bottom + pad_y)),
        )

    def _landmark_to_point(self, landmark, width, height, margin=0.08):
        if landmark is None:
            return None
        if landmark.x < -margin or landmark.x > 1.0 + margin:
            return None
        if landmark.y < -margin or landmark.y > 1.0 + margin:
            return None
        return (int(landmark.x * width), int(landmark.y * height))

    def _points_for_indices(self, landmarks, indices, width, height):
        points = []
        for index in indices:
            if index >= len(landmarks):
                continue
            point = self._landmark_to_point(landmarks[index], width, height)
            if point:
                points.append(point)
        return points

    def _rect_from_points(self, points, width, height):
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(width - 1, max(xs))),
            int(min(height - 1, max(ys))),
        )

    def _hull_from_points(self, points):
        if len(points) < 3:
            return None
        contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.convexHull(contour)

    def _hand_landmarks_to_rect(self, landmarks, width, height):
        points = []
        for lm in landmarks[:21]:
            point = self._landmark_to_point(lm, width, height)
            if point:
                points.append(point)

        if not points:
            return None

        return self._rect_from_points(points, width, height)

    def _is_probable_seltzer_sip(self, touching_points, landmarks, face_zone, width, height):
        if not self.seltzer_can_guard.get() or not touching_points:
            return False

        left, top, right, bottom = face_zone
        zone_w = max(1, right - left)
        zone_h = max(1, bottom - top)
        mouth_mid_y = (top + bottom) / 2
        sip_zone = self._expand_rect(face_zone, zone_w * 1.25, zone_h * 1.65, width, height)
        grip_zone = self._expand_rect(face_zone, zone_w * 2.25, zone_h * 3.25, width, height)

        def point_at(index):
            if index >= len(landmarks):
                return None
            return self._landmark_to_point(landmarks[index], width, height)

        hand_points = [point_at(index) for index in range(min(21, len(landmarks)))]
        hand_points = [point for point in hand_points if point is not None]
        if len(hand_points) < 5:
            return False

        hand_rect = self._rect_from_points(hand_points, width, height)
        wrist = point_at(0)
        palm_points = [point_at(index) for index in (0, 5, 9, 13, 17)]
        palm_points = [point for point in palm_points if point is not None]
        if wrist is None or len(palm_points) < 3:
            return False

        palm_x = sum(point[0] for point in palm_points) / len(palm_points)
        palm_y = sum(point[1] for point in palm_points) / len(palm_points)
        wrist_below_or_level = wrist[1] >= top - zone_h * 0.25
        palm_below_or_to_side = palm_y >= mouth_mid_y - zone_h * 0.35 or palm_x <= left or palm_x >= right
        if not wrist_below_or_level or not palm_below_or_to_side:
            return False

        if hand_rect and not self._rects_intersect(hand_rect, grip_zone):
            return False

        sip_points = 0
        for _, point in touching_points:
            if self._point_in_rect(point, sip_zone):
                sip_points += 1

        grip_points = 0
        support_points = 0
        for index in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18):
            point = point_at(index)
            if not point:
                continue
            if self._point_in_rect(point, grip_zone):
                grip_points += 1
            if self._point_in_rect(point, grip_zone) and not self._point_in_rect(point, face_zone):
                support_points += 1

        tip_near_mouth = 0
        for index in (4, 8, 12, 16):
            point = point_at(index)
            if point and self._point_in_rect(point, sip_zone):
                tip_near_mouth += 1

        sip_overlap_ratio = 0.0
        if hand_rect:
            overlap = self._rect_intersection(hand_rect, sip_zone)
            if overlap:
                sip_overlap_ratio = self._rect_area(overlap) / max(1.0, min(self._rect_area(hand_rect), self._rect_area(sip_zone)))

        if sip_points < 1 and tip_near_mouth < 1 and sip_overlap_ratio < 0.04:
            return False

        if grip_points >= 6 and support_points >= 3 and (tip_near_mouth >= 2 or sip_overlap_ratio >= 0.05):
            return True

        if tip_near_mouth >= 2 and support_points >= 2:
            return True

        if support_points >= 3 and sip_overlap_ratio >= 0.05:
            return True

        return False

    def _detect_drink_objects(self, mp_image, timestamp_ms, face_zone, width, height, hand_rect=None, force=False):
        if not self.drink_object_guard.get():
            return []
        if self.object_detector is None:
            self.load_object_detector_thread()
            return self.last_object_results
        if timestamp_ms <= self.last_object_timestamp_ms:
            return self.last_object_results

        self.object_frame_count += 1
        interval = max(2, int(self.drink_object_interval.get()))
        if not force and self.object_frame_count % interval != 0:
            return self.last_object_results

        object_results = self.object_detector.detect_for_video(mp_image, timestamp_ms)
        self.last_object_timestamp_ms = timestamp_ms
        near_zone = self._expand_rect(face_zone, width * 0.48, height * 0.36, width, height)
        held_zone = None
        if hand_rect:
            hand_w = max(1, hand_rect[2] - hand_rect[0])
            hand_h = max(1, hand_rect[3] - hand_rect[1])
            held_zone = self._expand_rect(
                hand_rect,
                max(width * 0.06, hand_w * 0.35),
                max(height * 0.06, hand_h * 0.35),
                width,
                height,
            )
        drinks = []

        for detection in object_results.detections:
            if not detection.categories:
                continue
            category = detection.categories[0]
            label = (category.category_name or "").lower()
            if label not in DRINK_OBJECT_LABELS:
                continue

            bbox = detection.bounding_box
            rect = (
                int(bbox.origin_x),
                int(bbox.origin_y),
                int(bbox.origin_x + bbox.width),
                int(bbox.origin_y + bbox.height),
            )
            frame_area = max(1, width * height)
            area_ratio = self._rect_area(rect) / frame_area
            if area_ratio > MAX_DRINK_AREA_RATIO:
                continue

            near_face = self._rects_intersect(rect, near_zone)
            held_by_hand = held_zone is not None and self._rects_intersect(rect, held_zone)
            accepted_as_held = held_by_hand and category.score >= HELD_DRINK_MIN_SCORE
            accepted_near_face_only = (
                held_zone is None
                and near_face
                and category.score >= FACE_ONLY_DRINK_MIN_SCORE
            )
            if accepted_as_held or accepted_near_face_only:
                drinks.append({
                    "label": label,
                    "score": category.score,
                    "rect": rect,
                    "source": "hand" if accepted_as_held else "face",
                })

        self.last_object_results = drinks
        if drinks:
            self.last_drink_seen_time = time.time()
        return drinks

    def _landmarks_to_rect(self, landmarks, width, height):
        xs = [lm.x * width for lm in landmarks]
        ys = [lm.y * height for lm in landmarks]
        return (
            max(0.0, min(xs)),
            max(0.0, min(ys)),
            min(float(width), max(xs)),
            min(float(height), max(ys)),
        )

    def _select_primary_face_landmarks(self, face_results, width, height):
        face_landmarks = getattr(face_results, "face_landmarks", None)
        if face_landmarks is not None:
            if not face_landmarks:
                return None
            faces = face_landmarks
        else:
            if not face_results.multi_face_landmarks:
                return None
            faces = [face.landmark for face in face_results.multi_face_landmarks]

        candidates = []
        frame_area = max(1, width * height)
        for landmarks in faces:
            if not landmarks:
                continue
            rect = self._landmarks_to_rect(landmarks, width, height)
            area = self._rect_area(rect)
            if area <= 0:
                continue

            center_x = (rect[0] + rect[2]) / 2
            center_y = (rect[1] + rect[3]) / 2
            center_distance = abs(center_x - (width / 2)) / width + abs(center_y - (height * 0.46)) / height
            score = (area / frame_area) * 4.0 - center_distance * 0.35

            if self.primary_face_rect is not None:
                score += self._rect_iou(rect, self.primary_face_rect) * 1.25

            candidates.append((score, landmarks, rect))

        if not candidates:
            return None

        _, landmarks, rect = max(candidates, key=lambda candidate: candidate[0])
        self.primary_face_rect = rect
        return landmarks

    def _build_face_model(self, face_results, width, height):
        landmarks = self._select_primary_face_landmarks(face_results, width, height)
        if not landmarks:
            return None
        face_left, face_top, face_right, face_bottom = self._landmarks_to_rect(landmarks, width, height)
        face_w = max(1.0, face_right - face_left)
        face_h = max(1.0, face_bottom - face_top)

        def y_at(index, fallback):
            if index < len(landmarks):
                return landmarks[index].y * height
            return fallback

        mouth_y = (y_at(13, face_top + face_h * 0.58) + y_at(14, face_top + face_h * 0.64)) / 2
        chin_y = y_at(152, face_bottom)

        face_points = self._points_for_indices(landmarks, FACE_OVAL_LANDMARKS, width, height)
        face_hull = self._hull_from_points(face_points)
        if face_hull is None:
            face_hull = self._hull_from_points([
                (face_left, face_top),
                (face_right, face_top),
                (face_right, face_bottom),
                (face_left, face_bottom),
            ])

        mouth_points = self._points_for_indices(landmarks, MOUTH_LANDMARKS, width, height)
        mouth_rect = self._rect_from_points(mouth_points, width, height)
        if mouth_rect is None:
            mouth_rect = (
                int(face_left + face_w * 0.28),
                int(face_top + face_h * 0.50),
                int(face_right - face_w * 0.28),
                int(face_top + face_h * 0.72),
            )

        lower_points = self._points_for_indices(landmarks, LOWER_FACE_LANDMARKS, width, height)
        lower_rect = self._rect_from_points(lower_points, width, height)
        pad = self.sensitivity.get() * face_w
        if lower_rect is None:
            lower_rect = (
                int(face_left + face_w * 0.10),
                int(mouth_y - face_h * 0.08),
                int(face_right - face_w * 0.10),
                int(chin_y + face_h * 0.12),
            )

        mouth_zone = self._expand_rect(
            mouth_rect,
            max(6, face_w * 0.12),
            max(5, face_h * 0.10),
            width,
            height,
        )
        lower_zone = self._expand_rect(
            lower_rect,
            max(8, pad * 0.85),
            max(6, pad * 0.65),
            width,
            height,
        )
        active_rect = self._expand_rect(
            (face_left, face_top, face_right, face_bottom),
            max(8, pad),
            max(8, pad),
            width,
            height,
        )

        return {
            "landmarks": landmarks,
            "face_rect": (face_left, face_top, face_right, face_bottom),
            "face_hull": face_hull,
            "active_rect": active_rect,
            "mouth_zone": mouth_zone,
            "lower_zone": lower_zone,
            "face_w": face_w,
            "face_h": face_h,
            "mouth_y": mouth_y,
            "chin_y": chin_y,
        }

    def _polygon_distance(self, point, contour, fallback_rect):
        if contour is not None and len(contour) >= 3:
            return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
        if self._point_in_rect(point, fallback_rect):
            return 1.0
        x, y = point
        left, top, right, bottom = fallback_rect
        dx = max(left - x, 0, x - right)
        dy = max(top - y, 0, y - bottom)
        return -float((dx * dx + dy * dy) ** 0.5)

    def _contact_padding(self, face_model):
        sensitivity = max(0.0, self.sensitivity.get())
        return max(7.0, face_model["face_w"] * (0.07 + sensitivity * 0.65))

    def _analyze_hand_contact(self, landmarks, face_model, width, height):
        padding = self._contact_padding(face_model)
        points = []
        near_points = []
        touching_points = []
        mouth_points = []
        beard_points = []
        score = 0.0
        inside_count = 0
        strong_finger_contact = False

        for index in range(min(21, len(landmarks))):
            point = self._landmark_to_point(landmarks[index], width, height)
            if not point:
                continue

            points.append(point)
            distance = self._polygon_distance(point, face_model["face_hull"], face_model["active_rect"])
            in_mouth = self._point_in_rect(point, face_model["mouth_zone"])
            in_lower = self._point_in_rect(point, face_model["lower_zone"])
            close_to_face = distance >= -padding

            if close_to_face or in_mouth or in_lower:
                weight = HAND_CONTACT_WEIGHTS.get(index, 0.75)
                proximity = 1.0 if distance >= 0 else max(0.0, 1.0 + (distance / max(1.0, padding)))
                point_score = weight * (1.10 + proximity)
                if distance >= 0:
                    point_score += 0.35
                    inside_count += 1
                if index in FINGER_TIP_LANDMARKS:
                    point_score += 0.40
                if in_mouth:
                    point_score += 0.30
                    mouth_points.append((index, point))
                if in_lower:
                    point_score += 0.45
                    beard_points.append((index, point))

                score += point_score
                touching_points.append((index, point))

                if index in FINGER_TIP_LANDMARKS or index in FINGER_EDGE_LANDMARKS:
                    if distance >= -padding * 0.45 or in_mouth or in_lower:
                        strong_finger_contact = True
            elif distance >= -padding * 1.8:
                near_points.append(point)

        hand_rect = self._hand_landmarks_to_rect(landmarks, width, height)
        if hand_rect:
            padded_face = self._expand_rect(
                face_model["active_rect"],
                padding * 0.35,
                padding * 0.35,
                width,
                height,
            )
            intersection = self._rect_intersection(hand_rect, padded_face)
            if intersection:
                overlap = self._rect_area(intersection) / max(1.0, min(self._rect_area(hand_rect), self._rect_area(padded_face)))
                if overlap >= 0.10:
                    score += min(2.3, overlap * 5.5)

        threshold = max(2.0, 2.9 - self.sensitivity.get() * 2.2)
        touching = score >= threshold or strong_finger_contact or (inside_count >= 2 and score >= 1.6)

        return {
            "touching": touching,
            "score": score,
            "points": points,
            "near_points": near_points,
            "touching_points": touching_points if touching else [],
            "candidate_points": touching_points,
            "mouth_points": mouth_points,
            "beard_points": beard_points,
            "mouth_contact": bool(mouth_points),
            "beard_contact": bool(beard_points),
            "hand_rect": hand_rect,
        }

    def _should_ignore_as_drink(self, contact, landmarks, face_model, mp_image, timestamp_ms, width, height, visible_drinks=None):
        if not contact["touching"]:
            return False, [], False

        drinks = list(visible_drinks or [])
        if self.drink_object_guard.get():
            object_drinks = self._detect_drink_objects(
                mp_image,
                timestamp_ms,
                face_model["mouth_zone"],
                width,
                height,
                hand_rect=contact["hand_rect"],
                force=True,
            )
            drinks = self._merge_drink_objects(drinks, object_drinks)

        if drinks and self._drink_suppresses_contact(drinks, contact, face_model):
            sip_only = all(drink.get("source") == "hand-pose" for drink in drinks)
            return True, drinks, sip_only

        if not self._mouth_dominant_contact(contact):
            return False, drinks, False

        if self._drink_memory_active():
            return True, drinks, False

        if self.seltzer_can_guard.get():
            probable_sip = self._is_probable_seltzer_sip(
                contact["candidate_points"],
                landmarks,
                face_model["mouth_zone"],
                width,
                height,
            )
            if probable_sip:
                self.last_sip_guard_time = time.time()
            if probable_sip or self._sip_guard_memory_active():
                rect = contact["hand_rect"] or face_model["mouth_zone"]
                if not drinks:
                    drinks = [{
                        "label": "seltzer grip",
                        "score": 1.0,
                        "rect": rect,
                        "source": "hand-pose",
                    }]
                return True, drinks, True

        return False, drinks, False

    def _mouth_dominant_contact(self, contact):
        mouth_count = len(contact.get("mouth_points", []))
        beard_count = len(contact.get("beard_points", []))
        return mouth_count > 0 and mouth_count > beard_count

    def _drink_suppresses_contact(self, drinks, contact, face_model):
        hand_rect = contact.get("hand_rect")
        mouth_zone = face_model["mouth_zone"]
        lower_zone = face_model["lower_zone"]
        mouth_dominant = self._mouth_dominant_contact(contact)

        for drink in drinks:
            rect = drink.get("rect")
            if not rect:
                continue

            source = drink.get("source")
            overlaps_mouth = self._rects_intersect(rect, mouth_zone)
            overlaps_lower = self._rects_intersect(rect, lower_zone)
            overlaps_hand = hand_rect is not None and self._rects_intersect(rect, hand_rect)

            if source == "hand-pose" and mouth_dominant and (overlaps_mouth or overlaps_hand):
                return True
            if source != "hand-pose" and overlaps_mouth and (overlaps_hand or source in {"hand", "face"}):
                return True
            if overlaps_lower and overlaps_hand and source in {"hand", "face"} and contact.get("mouth_contact"):
                return True

        return False

    def _make_sip_grip_marker(self, contact, face_model, label="can/cup grip"):
        rect = contact.get("hand_rect") or face_model["mouth_zone"]
        return {
            "label": label,
            "score": 1.0,
            "rect": rect,
            "source": "hand-pose",
        }

    def _merge_drink_objects(self, existing, incoming):
        merged = list(existing or [])
        for drink in incoming or []:
            rect = drink.get("rect")
            label = drink.get("label")
            duplicate = False
            for old in merged:
                old_rect = old.get("rect")
                if old.get("label") == label and rect and old_rect and self._rect_iou(rect, old_rect) >= 0.45:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(drink)
        return merged

    def _detect_visible_drink_candidate(self, contact, landmarks, face_model, mp_image, timestamp_ms, width, height):
        drinks = []

        hand_near_mouth = bool(contact["candidate_points"] or contact["near_points"] or contact["mouth_contact"])
        if self.drink_object_guard.get() and (hand_near_mouth or contact["hand_rect"]):
            object_drinks = self._detect_drink_objects(
                mp_image,
                timestamp_ms,
                face_model["mouth_zone"],
                width,
                height,
                hand_rect=contact["hand_rect"],
                force=False,
            )
            drinks = self._merge_drink_objects(drinks, object_drinks)

        if self.seltzer_can_guard.get() and hand_near_mouth:
            has_real_drink_box = any(drink.get("source") != "hand-pose" for drink in drinks)
            if not has_real_drink_box and self._mouth_dominant_contact(contact) and self._is_probable_seltzer_sip(
                contact["candidate_points"],
                landmarks,
                face_model["mouth_zone"],
                width,
                height,
            ):
                self.last_sip_guard_time = time.time()
                drinks.append(self._make_sip_grip_marker(contact, face_model))

        return drinks

    def _effective_skip_rate(self, base_skip):
        base_skip = max(1, int(base_skip))
        if base_skip == 1 or self.last_detection is None:
            return 1

        contact_score = self.last_detection.get("contact_score", 0.0)
        if self.last_detection.get("touching") or self.last_detection.get("drink_ignored"):
            return 1
        if contact_score >= 0.9:
            return 1
        if any(hand.get("near_points") for hand in self.last_detection.get("hands", [])):
            return 1
        return base_skip

    def _analyze_frame(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        face_results = self.face_mesh.detect_for_video(mp_image, timestamp_ms)
        hand_results = self.hands.detect_for_video(mp_image, timestamp_ms)
        face_model = self._build_face_model(face_results, width, height)

        detection = {
            "touching": False,
            "face": face_model,
            "face_zone": face_model["active_rect"] if face_model else None,
            "hands": [],
            "drink_ignored": False,
            "sip_guard_ignored": False,
            "drink_objects": [],
            "contact_score": 0.0,
            "message": "",
        }

        if face_model is None:
            detection["message"] = "No face detected"
            return detection

        hand_landmarks_list = getattr(hand_results, "hand_landmarks", None)
        handedness_list = getattr(hand_results, "handedness", None)
        if hand_landmarks_list is None:
            hand_landmarks_list = hand_results.multi_hand_landmarks
            handedness_list = hand_results.multi_handedness

        if not hand_landmarks_list:
            detection["message"] = "No hands detected"
            return detection

        min_score = self.conf_threshold.get()
        valid_hands = 0

        for hand_index, hand_landmarks in enumerate(hand_landmarks_list):
            score = 1.0
            label = "Hand"
            if handedness_list and hand_index < len(handedness_list):
                handedness = handedness_list[hand_index]
                classification = None
                if isinstance(handedness, (list, tuple)) and handedness:
                    classification = handedness[0]
                else:
                    classifications = getattr(handedness, "classification", None) or getattr(handedness, "classifications", None)
                    if classifications:
                        classification = classifications[0]
                if classification is not None:
                    score = getattr(classification, "score", 1.0)
                    label = (
                        getattr(classification, "category_name", None)
                        or getattr(classification, "display_name", None)
                        or getattr(classification, "label", "Hand")
                    )

            if min_score > 0 and score < min_score:
                continue

            valid_hands += 1
            landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
            contact = self._analyze_hand_contact(landmarks, face_model, width, height)
            ignored_points = []
            detection["contact_score"] = max(detection["contact_score"], contact["score"])
            visible_drinks = self._detect_visible_drink_candidate(
                contact,
                landmarks,
                face_model,
                mp_image,
                timestamp_ms,
                width,
                height,
            )
            detection["drink_objects"] = self._merge_drink_objects(detection["drink_objects"], visible_drinks)

            if contact["touching"]:
                ignore_as_drink, drink_objects, sip_ignored = self._should_ignore_as_drink(
                    contact,
                    landmarks,
                    face_model,
                    mp_image,
                    timestamp_ms,
                    width,
                    height,
                    visible_drinks=visible_drinks,
                )
                detection["drink_objects"] = self._merge_drink_objects(detection["drink_objects"], drink_objects)
                if ignore_as_drink:
                    detection["drink_ignored"] = True
                    detection["sip_guard_ignored"] = sip_ignored
                    ignored_points = [point for _, point in contact["touching_points"]]
                else:
                    detection["touching"] = True

            detection["hands"].append({
                "label": label,
                "score": score,
                "points": contact["points"],
                "near_points": contact["near_points"],
                "touching_points": [] if ignored_points else [point for _, point in contact["touching_points"]],
                "ignored_points": ignored_points,
            })

        if valid_hands == 0:
            detection["message"] = "Low hand confidence"
        elif detection["touching"]:
            detection["message"] = "Hand touching face"
        elif detection["drink_ignored"]:
            detection["message"] = "Sip ignored" if detection["sip_guard_ignored"] else "Drink ignored"
        elif detection["drink_objects"]:
            detection["message"] = "Drink near mouth"
        else:
            detection["message"] = "Hands clear"

        return detection

    def _draw_detection(self, frame, detection):
        h, _ = frame.shape[:2]
        touching = detection.get("touching", False)
        face = detection.get("face")
        face_zone = detection.get("face_zone")

        if face and face.get("face_hull") is not None:
            color = (0, 0, 255) if touching else (0, 180, 0)
            hull = face["face_hull"].astype(np.int32)
            cv2.polylines(frame, [hull], True, color, 2)
            lower = face.get("lower_zone")
            mouth = face.get("mouth_zone")
            if lower:
                cv2.rectangle(frame, (lower[0], lower[1]), (lower[2], lower[3]), (0, 170, 255), 1)
            if mouth:
                cv2.rectangle(frame, (mouth[0], mouth[1]), (mouth[2], mouth[3]), (255, 170, 0), 1)

        if face_zone:
            color = (0, 0, 255) if touching else (0, 180, 0)
            cv2.rectangle(frame, (face_zone[0], face_zone[1]), (face_zone[2], face_zone[3]), color, 1)

        for drink in detection.get("drink_objects", []):
            left, top, right, bottom = drink["rect"]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 120, 0), 2)
            cv2.putText(frame, f"{drink['label']} {drink['score']:.2f}", (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)

        for hand in detection.get("hands", []):
            for point in hand.get("near_points", []):
                cv2.circle(frame, point, 3, (255, 120, 0), 1)
            for point in hand["points"]:
                cv2.circle(frame, point, 4, (255, 200, 0), -1)
            for point in hand.get("ignored_points", []):
                cv2.circle(frame, point, 7, (255, 120, 0), -1)
            for point in hand["touching_points"]:
                cv2.circle(frame, point, 7, (0, 0, 255), -1)

        message = detection.get("message", "")
        if message:
            color = (0, 0, 255) if touching else (0, 150, 0)
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Streak: {self.touch_streak}/{max(1, self.required_streak.get())}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        cv2.putText(frame, f"Contact: {detection.get('contact_score', 0.0):.1f}", (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

    def process_video(self):
        try:
            if not self.is_tracking or not self.cap:
                return

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                self.stop_tracking()
                return
                
            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - self.last_frame_time) if self.last_frame_time > 0 else 0
            self.last_frame_time = current_time

            # Frame Skipping
            self.frame_count += 1
            try:
                base_skip_rate = int(self.frame_skip.get())
                if base_skip_rate < 1: base_skip_rate = 1
            except:
                base_skip_rate = 2
            skip_rate = self._effective_skip_rate(base_skip_rate)
            
            # Always display the frame, but only run inference on throttled frames.
            run_inference = self.last_detection is None or (self.frame_count % skip_rate == 0)

            annotated_frame = frame  # annotate in place; full-resolution copy is unnecessary

            # If we run inference, update the stored hand/face detection.
            if run_inference:
                try:
                    self.last_detection = self._analyze_frame(frame)
                    touching = self.last_detection["touching"]
                    self.contact_score = self.last_detection.get("contact_score", 0.0)

                    if touching:
                        self.touch_streak += 1
                    else:
                        self.touch_streak = 0
                        self.in_alert_state = False

                    required = max(1, self.required_streak.get())
                    if self.touch_streak >= required:
                        self.in_alert_state = True
                        self.alert_end_time = time.time() + self.alert_duration.get()
                except Exception as e:
                    print(f"Inference error: {e}")

            if self.last_detection:
                self._draw_detection(annotated_frame, self.last_detection)

            if time.time() < self.alert_end_time:
                self.show_alert()
            else:
                self.hide_alert()

            # Draw FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert to Tkinter
            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

            # Preview FPS is capped, and inference is further throttled by frame_skip.
            self.root.after(self.preview_delay_ms, self.process_video)
        
        except Exception as e:
            print(f"CRITICAL ERROR in process_video: {e}")
            self.stop_tracking()

if __name__ == "__main__":
    root = tk.Tk()
    def on_closing():
        if app.cap: app.cap.release()
        app.close_detectors()
        root.destroy()
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    app = BeardTracker(root)
    root.mainloop()
