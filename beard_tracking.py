import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import sys
import os
from screeninfo import get_monitors

class BeardTrackerYOLO:
    def __init__(self, root):
        self.root = root
        self.root.title("Beard Picking Tracker (YOLO)")
        self.root.geometry("1000x800")
        
        # State
        self.is_tracking = False
        self.is_alert_active = False
        self.alert_count = 0
        self.cap = None
        self.model = None
        self.alert_end_time = 0
        self.px_shoulder_width = 0 # Cache for shoulder width
        self.flash_state = False # For flashing text
        self.flash_job = None # To cancel flashing
        
        # Performance/Optimization State
        self.last_results = None
        self.frame_count = 0
        self.last_frame_time = 0

        # Settings
        self.sensitivity = tk.DoubleVar(value=0.5) # Default ration 0.5
        self.alert_duration = tk.DoubleVar(value=1.0) # Default duration 1.0s
        self.frame_skip = tk.IntVar(value=3) # Process every Nth frame

        # UI Elements Container
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, fill=tk.X)

        # Buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=20)
        
        self.start_btn = tk.Button(btn_frame, text="Start Tracking", command=self.start_tracking, font=("Arial", 14), bg="green", fg="white")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop Tracking", command=self.stop_tracking, state=tk.DISABLED, font=("Arial", 14), bg="red", fg="white")
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Slider
        slider_frame = tk.Frame(control_frame)
        slider_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        tk.Label(slider_frame, text="Sensitivity (Distance to Trigger)", font=("Arial", 12)).pack()
        # Scale from 0.2 (Hard to trigger, close) to 2.0 (Easy to trigger, far)
        self.slider = tk.Scale(slider_frame, from_=0.2, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.sensitivity, length=300)
        self.slider.pack(fill=tk.X)
        
        tk.Label(slider_frame, text="Alert Duration (Seconds)", font=("Arial", 12)).pack(pady=(10, 0))
        self.duration_slider = tk.Scale(slider_frame, from_=0.5, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.alert_duration, length=300)
        self.duration_slider.pack(fill=tk.X)
        
        tk.Label(slider_frame, text="Performance: Skip Frames (1 = Max CPU, 10 = Eco)", font=("Arial", 12)).pack(pady=(10, 0))
        self.skip_slider = tk.Scale(slider_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL, variable=self.frame_skip, length=300)
        self.skip_slider.pack(fill=tk.X)
        
        self.status_label = tk.Label(root, text="Status: Idle (Model Loading...)", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.count_label = tk.Label(root, text="Alerts: 0", font=("Arial", 12, "bold"), fg="blue")
        self.count_label.pack(pady=2)

        # Video Label
        self.video_label = tk.Label(root)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Alert Windows (One per monitor)
        self.alert_windows = []
        self.alert_labels = [] # To reference text for flashing
        self.vignette_images = [] # Keep references to prevent GC
        self.create_alert_windows()

        # Load Model
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
        Mask = Mask ** 3 # Cube it to push the red to the edges
        
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
        # Black is the transparent key.
        # Center of vignette is black -> Transparent window hole.
        # Edges are Red -> Visible Red.
        win.configure(bg='black')
        try:
             win.attributes("-transparentcolor", "black")
             win.attributes("-alpha", 0.75) # Semi-transparent blend
        except:
            print("Transparency not supported on this platform/config")

        # Generate Background
        pil_img = self.generate_vignette(w, h)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.vignette_images.append(tk_img) # Keep ref
        
        bg_label = tk.Label(win, image=tk_img, bg='black')
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Text Label
        # Font increased to 150.
        # fg="red" initially. 
        # bg="black" matches transparent key.
        label = tk.Label(win, text="STOP PICKING", font=("Arial", 150, "bold"), fg="red", bg="black")
        label.place(relx=0.5, rely=0.5, anchor="center")
        self.alert_labels.append(label)
        
        # FORCE QUIT BUTTON (Top Right)
        quit_btn = tk.Button(win, text="FORCE QUIT", command=self.force_quit, font=("Arial", 12, "bold"), bg="white", fg="red")
        quit_btn.place(relx=0.98, rely=0.03, anchor="ne")

        win.bind("<Escape>", lambda e: self.hide_alert())
        self.alert_windows.append(win)

    def force_quit(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()
        sys.exit(0)

    def load_model_thread(self):
        def _load():
            try:
                self.model = YOLO('yolov8n-pose.pt') 
                self.root.after(0, lambda: self.status_label.config(text="Status: Ready"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Error loading model: {e}"))
        
        Thread(target=_load, daemon=True).start()

    def start_tracking(self):
        if self.is_tracking or self.model is None: return
        self.is_tracking = True
        self.alert_count = 0
        self.count_label.config(text=f"Alerts: {self.alert_count}")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Tracking...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video source")
                
            # Set Resolution to 640x480 for performance (Commented out for debugging)
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.process_video()
        except Exception as e:
            print(f"Error starting tracking: {e}")
            self.stop_tracking()

    def stop_tracking(self):
        self.is_tracking = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
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
            for label in self.alert_labels:
                label.config(fg="red")

    def flash_alert_loop(self):
        if not self.is_alert_active: return
        
        # Toggle Color
        # Flashing Red <-> Black
        # Black text on Black background = Invisible
        # This creates a blinking effect.
        new_color = "black" if self.flash_state else "red"
        
        for label in self.alert_labels:
            label.config(fg=new_color)
            
        self.flash_state = not self.flash_state
        # Schedule next flash (200ms)
        self.flash_job = self.root.after(200, self.flash_alert_loop)

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
                skip_rate = int(self.frame_skip.get())
                if skip_rate < 1: skip_rate = 1
            except:
                skip_rate = 3
            
            # Always display the frame, but only run inference if it's the right frame
            # OR if we haven't run inference yet (results is None)
            run_inference = (self.frame_count % skip_rate == 0)

            annotated_frame = frame.copy()
            
            # If we run inference, update the stored results
            if run_inference:
                try:
                    # OPTIMIZATION: imgsz=256 drastically reduces CPU usage compared to default 640
                    self.last_results = self.model(frame, verbose=False, imgsz=256)
                except Exception as e:
                    print(f"Inference error: {e}")
            
            # If we have results (either fresh or stale), draw them
            if self.last_results:
                results = self.last_results
                touching = False
                
                # Analyze Keypoints
                if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
                    kpts = results[0].keypoints.data[0].cpu().numpy() # First person
                    
                    # Indices: 0:Nose, 5:L_Shoulder, 6:R_Shoulder, 9:L_Wrist, 10:R_Wrist
                    if kpts.shape[0] > 10:
                        nose = kpts[0][:2]
                        l_sh = kpts[5][:2]
                        r_sh = kpts[6][:2]
                        l_wrist = kpts[9][:2]
                        r_wrist = kpts[10][:2]
                        
                        # Confidence check (Lowered to 0.3)
                        CONF_THRESH = 0.3
                        
                        # Update Shoulder Width if visible
                        if kpts[5][2] > CONF_THRESH and kpts[6][2] > CONF_THRESH:
                            self.px_shoulder_width = np.linalg.norm(l_sh - r_sh)
                            
                        if self.px_shoulder_width > 0:
                            # Normalize wrist distances
                            dist_l = np.linalg.norm(nose - l_wrist) / self.px_shoulder_width
                            dist_r = np.linalg.norm(nose - r_wrist) / self.px_shoulder_width
                            
                            threshold = self.sensitivity.get()
                            
                            # Draw Meter Background
                            h, w = frame.shape[:2]
                            meter_w = 30
                            meter_h = 200
                            meter_x = w - 50
                            meter_y = 50
                            
                            # Calculate closest hand ratio for the meter (inverted: 0 is dangerous)
                            conf_l = kpts[9][2]
                            conf_r = kpts[10][2]
                            min_dist = min(dist_l if conf_l > CONF_THRESH else 999, dist_r if conf_r > CONF_THRESH else 999)
                            
                            # Visualization: Draw lines
                            if conf_l > CONF_THRESH:
                                color = (0, 255, 0) if dist_l > threshold else (0, 0, 255)
                                cv2.line(annotated_frame, (int(nose[0]), int(nose[1])), (int(l_wrist[0]), int(l_wrist[1])), color, 2)
                            
                            if conf_r > CONF_THRESH:
                                color = (0, 255, 0) if dist_r > threshold else (0, 0, 255)
                                cv2.line(annotated_frame, (int(nose[0]), int(nose[1])), (int(r_wrist[0]), int(r_wrist[1])), color, 2)

                            # Trigger
                            if min_dist < threshold:
                                touching = True
                                self.alert_end_time = time.time() + self.alert_duration.get()

                            # Draw Visual Meter
                            # Draw Background (Grey)
                            cv2.rectangle(annotated_frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (200, 200, 200), -1)
                            
                            # Bar Height: Safety Meter (Full = Safe/Far, Empty = Danger/Close)
                            # Max range assumed ~2.0 shoulder widths
                            display_val = max(0, min(1.0, min_dist / 2.0))
                            bar_h = int(display_val * meter_h)
                            
                            # Color gradient based on closeness
                            bar_color = (0, 255, 0) # Green (Safe)
                            if min_dist < threshold * 1.5: bar_color = (0, 255, 255) # Yellow
                            if min_dist < threshold: bar_color = (0, 0, 255) # Red (Danger)

                            # Draw Fill
                            cv2.rectangle(annotated_frame, (meter_x, meter_y + meter_h - bar_h), (meter_x + meter_w, meter_y + meter_h), bar_color, -1)
                            # Draw Border
                            cv2.rectangle(annotated_frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (0, 0, 0), 2)
                            
                            cv2.putText(annotated_frame, f"{min_dist:.2f}", (meter_x - 10, meter_y + meter_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            
                            # Draw threshold line on meter
                            # Convert threshold (distance) to y position
                            # 0 distance = BOTTOM (y + h)
                            # 2.0 distance = TOP (y)
                            thresh_y = meter_y + meter_h - int((threshold / 2.0) * meter_h)
                            cv2.line(annotated_frame, (meter_x - 5, thresh_y), (meter_x + meter_w + 5, thresh_y), (0, 0, 255), 2)
                            
                            # Debug Info
                            cv2.putText(annotated_frame, f"L Conf: {conf_l:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(annotated_frame, f"R Conf: {conf_r:.2f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        else:
                            cv2.putText(annotated_frame, "Shoulders not detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if touching or time.time() < self.alert_end_time:
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

            self.root.after(10, self.process_video)
        
        except Exception as e:
            print(f"CRITICAL ERROR in process_video: {e}")
            self.stop_tracking()

if __name__ == "__main__":
    root = tk.Tk()
    def on_closing():
        if app.cap: app.cap.release()
        root.destroy()
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    app = BeardTrackerYOLO(root)
    root.mainloop()