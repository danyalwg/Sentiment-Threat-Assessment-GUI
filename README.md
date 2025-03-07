# Sentiment & Threat Assessment GUI

## Overview
This project is a **Sentiment & Threat Assessment GUI** that analyzes facial emotions in real-time video feeds, images, or webcam streams. It calculates a **threat level** based on detected emotions using a **Residual Masking Network (RMN)** and visualizes the threat trends using **Matplotlib**. The graphical interface is built using **PyQt5**.

## Features
- **Facial Emotion Recognition** using RMN
- **Threat Level Calculation** based on emotion contributions
- **Supports Images, Videos, and Live Webcam Feeds**
- **Real-time Graph Plotting** of threat level over time
- **Customizable Parameters** (scaling, bias, smoothing)
- **Exportable Threat Reports**

---
## How It Works

### 1. **Facial Emotion Detection**
   - Uses `RMN` (Residual Masking Network) to classify facial expressions into:
     - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 2. **Threat Level Calculation**
   - Computes a weighted threat score using predefined weights for each emotion.
   - Applies **smoothing, scaling, and bias** for better stability.
   - **Synergy Bonus:** Adjusts threat if multiple people show negative emotions.
   - Categorized into: **Low, Medium, High, Critical**

### 3. **Graphical Display & Analysis**
   - **Video Feed:** Displays annotated image/video with detected faces.
   - **Real-time Plot:** Shows threat level trend over time.
   - **Summary Report:** Provides statistics on average threat levels and emotion distribution.

---
## Installation & Setup

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.7+

### **2. Install Dependencies**
Run the following command to install all required packages:
```
pip install opencv-python PyQt5 matplotlib rmn
```

### **3. Running the GUI**
To launch the application, execute:
```
python gui.py
```

---
## Usage Guide

### **1. Load an Image or Video**
- Click **Load File** and select an image (`.jpg, .png`) or video (`.mp4, .avi`).
- The system will analyze the content and display the **threat level**.

### **2. Live Webcam Analysis**
- Click **Open Webcam** to start live detection.
- Click **Stop Feed** to halt detection.

### **3. Understanding the Output**
- **Threat Level:** Displayed on-screen in real-time.
- **Graph:** Shows how threat levels change over time.
- **Report Summary:** Highlights key statistics, including average and peak threat.

### **4. Export Threat Report**
- Click **Export Report** to save the analysis as a `.txt` file.

---
## Directory Structure
```
Sentiment-Threat-Assessment-GUI/
├── gui.py              # Main application script
└── README.md           # Documentation
```

---
## Configuration
The following parameters can be adjusted in `gui.py`:
- **scaling:** Adjusts threat level intensity.
- **bias:** Offsets the threat calculation.
- **smoothing:** Controls how much previous frames influence the current threat level.

Modify them here:
```
self.scaling = 1.0
self.bias = 1.0
self.smoothing = 0.9
```

---
## Troubleshooting
### **1. Application Crashes on Launch**
- Ensure you have all dependencies installed.

### **2. Webcam Not Working**
- Try restarting your webcam and ensure no other applications are using it.
- Run `python gui.py` from the terminal to check for error messages.

### **3. Threat Level Seems Too High/Low**
- Adjust the `scaling`, `bias`, and `smoothing` parameters.

---
## Future Improvements
- Implement **Multi-Factor Threat Analysis** (e.g., voice tone, body posture).
- Add **Customizable Emotion Weights** per use case.
- Improve **Graphical UI Enhancements** for better visualization.

---
## Contribution
Contributions are welcome! Feel free to:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Added new feature"`).
4. Push and create a pull request.

---
## License
This project is licensed under the MIT License.

---
## Author
**GitHub:** [@danyalwg](https://github.com/danyalwg)
