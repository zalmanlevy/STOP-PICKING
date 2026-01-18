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
        self.cap = None
        self.model = None
        self.alert_end_time = 0
        self.px_shoulder_width = 0 # Cache for shoulder width

        # Settings
        self.sensitivity = tk.DoubleVar(value=0.5) # Default ration 0.5

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
        
        self.status_label = tk.Label(root, text="Status: Idle (Model Loading...)", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Video Label
        self.video_label = tk.Label(root)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Alert Windows (One per monitor)
        self.alert_windows = []
        self.create_alert_windows()

        # Load Model
        self.load_model_thread()

    def create_alert_windows(self):
        # Clean up existing if any
        for win in self.alert_windows:
            win.destroy()
        self.alert_windows = []

        try:
            monitors = get_monitors()
            for m in monitors:
                win = tk.Toplevel(self.root)
                win.withdraw()
                # Geometry: widthxheight+x+y
                win.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
                win.overrideredirect(True) # Remove interactions/title bar
                win.attributes("-topmost", True)
                win.configure(bg='red')
                
                label = tk.Label(win, text="STOP PICKING", font=("Arial", 100, "bold"), fg="white", bg="red")
                label.place(relx=0.5, rely=0.5, anchor="center")
                
                # Bind escape to hide on all windows
                win.bind("<Escape>", lambda e: self.hide_alert())
                
                self.alert_windows.append(win)
        except Exception as e:
            print(f"Error checking monitors: {e}. Fallback to single window logic.")
            win = tk.Toplevel(self.root)
            win.withdraw()
            win.attributes("-fullscreen", True)
            win.attributes("-topmost", True)
            win.configure(bg='red')
            label = tk.Label(win, text="STOP PICKING", font=("Arial", 100, "bold"), fg="white", bg="red")
            label.place(relx=0.5, rely=0.5, anchor="center")
            win.bind("<Escape>", lambda e: self.hide_alert())
            self.alert_windows.append(win)

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
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Tracking...")
        
        self.cap = cv2.VideoCapture(0)
        self.process_video()

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
            for win in self.alert_windows:
                win.deiconify()
                win.attributes("-topmost", True) # Reinforce on top

    def hide_alert(self):
        if self.is_alert_active:
            self.is_alert_active = False
            for win in self.alert_windows:
                win.withdraw()

    def process_video(self):
        if not self.is_tracking or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_tracking()
            return

        # Inference
        results = self.model(frame, verbose=False)
        annotated_frame = frame.copy()
        
        touching = False
        
        # Analyze Keypoints
        if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
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
                        self.alert_end_time = time.time() + 1.0

                    # Draw Visual Meter
                    cv2.rectangle(annotated_frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (200, 200, 200), -1)
                    
                    # Bar Height (Inverse: Closer = Higher Bar)
                    display_val = max(0, min(1.0, (2.0 - min_dist) / 2.0))
                    bar_h = int(display_val * meter_h)
                    
                    # Color gradient based on closeness
                    bar_color = (0, 255, 0) # Green
                    if min_dist < threshold * 1.5: bar_color = (0, 255, 255) # Yellow
                    if min_dist < threshold: bar_color = (0, 0, 255) # Red

                    cv2.rectangle(annotated_frame, (meter_x, meter_y + meter_h - bar_h), (meter_x + meter_w, meter_y + meter_h), bar_color, -1)
                    cv2.rectangle(annotated_frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (0, 0, 0), 2)
                    cv2.putText(annotated_frame, f"{min_dist:.2f}", (meter_x - 10, meter_y + meter_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Draw threshold line on meter
                    thresh_y = meter_y + meter_h - int(((2.0 - threshold) / 2.0) * meter_h)
                    cv2.line(annotated_frame, (meter_x - 5, thresh_y), (meter_x + meter_w + 5, thresh_y), (0, 0, 255), 2)
                    
                    # Debug Info
                    cv2.putText(annotated_frame, f"L Conf: {conf_l:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"R Conf: {conf_r:.2f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "Shoulders not detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Convert to Tkinter
        image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        if touching or time.time() < self.alert_end_time:
            self.show_alert()
        else:
            self.hide_alert()

        self.root.after(10, self.process_video)

if __name__ == "__main__":
    root = tk.Tk()
    def on_closing():
        if app.cap: app.cap.release()
        root.destroy()
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    app = BeardTrackerYOLO(root)
    root.mainloop()