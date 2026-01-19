# Face Touch Tracker (Don't Touch Your Face!)

## ⚠️ Read This First
I originally built this tool just for myself to help break the habit of picking my face/beard while working. I'm putting it out here in case anyone else finds it useful!

Feel free to use it, but please keep in mind this is a personal project—it is provided "as is."

## 🔒 Privacy & Security
**This app runs 100% locally on your computer.**
It does **not** send your video feed or data to the internet.

* **Proof:** You can literally turn off your Wi-Fi/Internet connection, and the app will still work perfectly.

## 📥 How to Download (Important!)
**You are currently looking at the main code, but the ready-to-run version is on a different "Branch."**

1.  Look at the screenshot below. You need to switch from **main** to **Final App**.
2.  Click the branch button (top left) and select **Final App**.
3.  Once the page reloads, click the green **Code** button and select **Download ZIP**.
4.  Unzip the folder onto your desktop.

<img width="871" height="359" alt="image" src="https://github.com/user-attachments/assets/de2d9c60-a787-4b66-b903-83e42d257ced" />

## 🚀 How to Run the App
I've included a few ways to run this, depending on what you prefer.

### Option 1: The Easiest Way (.exe)
1.  Open the folder you downloaded.
2.  Go into the folder named **`dist`**.
3.  Double-click the **.exe** application file inside.
4.  That's it! The webcam should start.

### Option 2: The Batch File
If you don't trust me (lol) and prefer not to run the .exe directly, you can run the batch file straight.
1.  Go to the main folder.
2.  Double-click the file named `run_beard_tracking.bat`.

### Option 3: For Developers
If you know what you are doing, you can run the source code directly through your terminal/command line in the main directory.

## ⚙️ Settings Explained
Once the app is running, you will see a few controls:
* **Sensitivity:** The higher this is, the more often the alarm will run (it becomes more sensitive to you touching your face).
* **Alert Duration:** This controls how many seconds the blaring alarm and red screen stay on for.
* **Performance Slider:** The higher you set this number, the **less** CPU/computer power the app uses (slide this up if your computer is lagging).

## 🛑 How to Stop It
* **If the alarm IS blaring:** Click **"Force Quit"** in the top right corner.
* **If the alarm is NOT blaring:** You can just click the standard **X** on the window to close the application.
