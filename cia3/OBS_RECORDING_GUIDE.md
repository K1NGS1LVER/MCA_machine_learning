# 🎥 OBS Studio Recording Guide (Face Cutout PiP Setup)

This guide shows you step-by-step how to record your 3-minute video presentation in **OBS Studio** on macOS with your face in a rounded/circular cutout picture-in-picture (PiP) window.

---

## 🛠️ Step 1: Launch Presentation Slides
1. Open [`slides.html`](file:///Users/dan/projects/pythonVishal/machine_learning/cia3/slides.html) in your browser (Chrome, Arc, or Safari).
2. Press **`F`** to toggle **Fullscreen mode** (or use your browser's full-screen shortcut `Cmd + Shift + F`).
3. Press **`Arrow Right`** or **`Spacebar`** to advance slides, and **`Arrow Left`** to go back.
4. *Notice the bottom-right corner:* The slides have been intentionally designed with empty space on the bottom-right specifically so your camera cutout doesn't cover any text, charts, or tables!

---

## 📹 Step 2: Configure OBS Studio Scene

### 1. Set Base Canvas Resolution
- Go to **OBS Studio Settings** > **Video**.
- **Base (Canvas) Resolution:** `1920x1080`
- **Output (Scaled) Resolution:** `1920x1080`
- **Common FPS Values:** `60` or `30`

### 2. Add Screen Capture (Slide Background)
1. In the **Sources** dock at the bottom, click the **`+`** icon.
2. Select **macOS Screen Capture** (or **Window Capture**).
3. Set Method to **Window Capture** and pick your browser window showing `slides.html`.
4. Click **OK**. Resize if needed to fill the full 1080p canvas.

### 3. Add Video Capture Device (Your Face)
1. Click **`+`** in the Sources dock.
2. Select **Video Capture Device**.
3. Choose your built-in FaceTime HD Camera / External USB Webcam.
4. Drag and resize the camera box and position it at the **bottom-right corner** of the screen.

---

## 🎭 Step 3: Create the Rounded/Circular Face Cutout

You have two easy ways to make your webcam a clean cutout:

### Method A: Rounded Mask Filter (Recommended)
1. Right-click your **Video Capture Device** source in OBS > select **Filters**.
2. Under **Effect Filters**, click **`+`** > select **Image Mask/Blend**.
3. **Type:** `Alpha Mask (Alpha Channel)` or `Alpha Mask (Color Channel)`.
4. Create or download a simple black/white circle PNG (or rounded rectangle) and browse to it.
5. Click **Close**. Your camera is now a circle or rounded card!

### Method B: Native Alt-Crop & Border
1. In the OBS preview, hold down the **`Option` key (Alt)** on macOS while dragging the edges of your camera box to crop it into a square.
2. Right-click the camera source > **Filters** > click **`+`** > select **Crop/Pad** or **Border** (via StreamFX/plugins) if installed.

---

## 🎙️ Step 4: Audio & Mic Setup
1. In **Audio Mixer**, ensure your microphone source is active.
2. Click the gear icon next to your Mic > **Filters** > add:
   * **Noise Suppression** (RNNoise - high quality)
   * **Gain** (+2 to +4 dB if your voice is soft)
3. Do a quick 5-second test recording to verify crisp audio.

---

## ⏱️ Step 5: Recording Workflow
1. Start the presentation timer inside [`slides.html`](file:///Users/dan/projects/pythonVishal/machine_learning/cia3/slides.html) by clicking **"Start Pitch"**.
2. Press **Start Recording** in OBS Studio (or set a hotkey like `Cmd + Shift + R`).
3. Follow the spoken script from [`PITCH_SCRIPT_3MIN.md`](file:///Users/dan/projects/pythonVishal/machine_learning/cia3/PITCH_SCRIPT_3MIN.md) on screen.
4. At exactly 3:00, conclude with *"Thank you for your time"* and press **Stop Recording**.
5. Your finalized MP4/MKV video will be saved in your macOS Movies / Videos folder!
