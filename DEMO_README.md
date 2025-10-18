# 🎮 PixelsToPlayers Demo

A comprehensive demonstration of all PixelsToPlayers components working together.

## ⚙️ Requirements

- **Python 3.8+**
- **Poetry** (for dependency management)
- **Webcam/camera** (built-in or external)
- **Good lighting** (for face detection)
- **Screen recording permissions** (macOS will ask)

## 📦 Install Dependencies with Poetry

This project uses Poetry to manage dependencies. Here's how to set everything up:

### 1. Install Poetry (if you don't have it)
```bash
# Via pip
pip install poetry
```

### 2. Install Project Dependencies
```bash
# Navigate to the project root
cd /path/to/PixelsToPlayers-PythonHelper

# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 3. Run the Demo
```bash
# Navigate to the demo location
cd src/pixels_to_players

# Run the demo
python3 demo_test.py
```

## 🚀 Quick Start (After Setup)

```bash
# Navigate to the project directory
cd /path/to/PixelsToPlayers-PythonHelper/src/pixels_to_players

# Run the demo
python3 demo_test.py
```

## 📋 What You'll See

The demo includes **5 main components**:

| Component | What It Does |
|-----------|--------------|
| 🎥 **Webcam** | Shows your face with AI mesh overlay |
| 👁️ **Gaze Tracking** | Tracks where you're looking in real-time |
| 🎯 **Calibration** | Maps your eye movements to screen coordinates |
| 📹 **Screen Recording** | Captures your screen while tracking gaze |
| 🔄 **Integrated Demo** | Everything working together |

## 🎯 Demo Menu

When you run the demo, you'll see this menu:

```
1. Test Webcam (Face Mesh Detection)
2. Test Gaze Tracking  
3. Run Calibration
4. Test Screen Recording
5. Integrated Demo (Gaze + Screen Recording)
6. Run All Tests
0. Exit
```

## 📁 What Gets Created

The demo saves files to help you see what's happening:

- `demo_calibration_data.json` - Your eye calibration data
- `demo_recordings/` - Screen recordings
- `gaze_data_*.json` - Gaze tracking samples


## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Close other apps using the camera |
| Face mesh not showing | Check lighting, make sure face is visible |
| Screen recording fails | Grant screen recording permissions |
| Calibration not working | Look directly at each calibration point |

## 🎯 Key Features

- ✅ **Real-time face mesh detection**
- ✅ **Accurate gaze tracking** 
- ✅ **5-point calibration system**
- ✅ **Screen recording with gaze data**
- ✅ **Interactive menu system**
- ✅ **Data collection and storage**
---

*This demo showcases the core functionality that will power the final PixelsToPlayers application!*