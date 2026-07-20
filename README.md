# Face Touch Tracker (Don't Touch Your Face!)

## ⚠️ Read This First
I originally built this tool just for myself to help break the habit of picking my face/beard while working. I'm putting it out here in case anyone else finds it useful!

**Please note that I used Gemini in AntiGravity to make this entirely and I barely know how to code :)**

Feel free to use it, but please keep in mind this is a personal project—it is provided "as is."

## 🔒 Privacy & Security
**This app runs 100% locally on your computer.**
It does **not** send your video feed or data to the internet.
On first setup, it may download the official MediaPipe hand and face detector model files into the local `models` folder. After that, detection runs locally.

* **Proof:** You can literally turn off your Wi-Fi/Internet connection, and the app will still work perfectly.

## 📥 How to Download

### Option 1: Download the App (EXE)
If you just want to run the app without messing with code:
1.  Look for the **Releases** section on the right-hand side of this page.
2.  Click to download **BeardTracker App**.
3.  Unzip the folder onto your desktop.

### Option 2: Get the Source Code (Important!)
**If you are looking for the code, you must switch branches.** The main branch is empty; the real code is on the "Final App" branch.

1.  Look at the screenshot below. You need to switch from **main** to **Final App**.
2.  Click the branch button (top left) and select **Final App**.
3.  Once the page reloads, click the green **Code** button and select **Download ZIP**.

<img width="871" height="359" alt="image" src="https://github.com/user-attachments/assets/de2d9c60-a787-4b66-b903-83e42d257ced" />

## 🚀 How to Run the App
I've included a few ways to run this, depending on what you prefer.

### Method 1: The Easiest Way (.exe)
1.  Open the folder you downloaded from **Releases**.
2.  Double-click the **.exe** application file inside.
3.  That's it! The webcam should start.

### Method 2: The Batch File
If you don't trust me (lol) and prefer not to run the .exe directly, you can run the batch file straight from the source code.
1.  Go to the main folder.
2.  Double-click the file named `run_beard_tracking.bat`.

### Method 3: For Developers
If you know what you are doing, you can run the source code directly through your terminal/command line in the main directory.

## ⚙️ Settings Explained
Once the app is running, you will see a few controls:
* **Sensitivity:** Expands or shrinks the face-contact zone. Higher values catch more near-face touches.
* **Alert Duration:** This controls how many seconds the blaring alarm and red screen stay on for.
* **Performance Slider:** The higher you set this number, the less often the app analyzes a frame, which lowers CPU usage. The app temporarily returns to full analysis speed when a hand gets near your face.
* **Hand Confidence Filter:** Optional. Leave it at `0` unless you need to ignore unstable hand detections.
* **Drink Object Detector:** Runs intermittently near your mouth/hand. Looks for lightweight COCO drink-like objects such as bottles and cups overlapping the active hand. It only suppresses mouth-area contact; beard/chin contact still alerts.
* **Drink Memory:** Keeps a recently detected drink active very briefly so one missed frame does not immediately alert mid-sip.
* **Seltzer Can Sip Guard:** Lightweight hand-landmark fallback for cans that the object detector misses. It draws a `can/cup grip` marker near the mouth and only suppresses mouth-area contact when the wrist, palm, and finger landmarks look like a drinking grip.
* **Sip Guard Memory:** Keeps the seltzer-can sip guard active very briefly between frames.
* **Capture Resolution:** Lower resolutions use less CPU. The default is 480x360.

The app tracks the primary face by favoring the largest, closest, and most stable face in view. It scores contact against the full face outline, mouth area, lower face, fingertips, finger edges, and palm overlap instead of relying only on wrist distance.

## 🛑 How to Stop It
* **If the alarm IS blaring:** Click **"Force Quit"** in the top right corner.
* **If the alarm is NOT blaring:** You can just click the standard **X** on the window to close the application.
