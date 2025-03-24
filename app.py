import cv2
import streamlit as st
from gtts import gTTS
import os
import time
import base64
from collections import deque

# Create the alerts folder if it doesn't exist
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# Queue to store missing PPE alerts
alert_queue = deque()

# Function to generate voice alerts
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("alerts/alert.mp3")
    
    # Read the audio file and encode it in base64
    with open("alerts/alert.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # Autoplay the audio using JavaScript
    autoplay_audio = f"""
    <audio id="audio" autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    <script>
        var audio = document.getElementById("audio");
        audio.play();
    </script>
    """
    st.components.v1.html(autoplay_audio, height=0)

# Function to process the alert queue
def process_alerts():
    if alert_queue:
        # Combine missing items into a single sentence
        missing_items = list(alert_queue)
        if len(missing_items) == 1:
            alert_text = f"Not wearing {missing_items[0]}."
        else:
            alert_text = "Not wearing " + ", ".join(missing_items[:-1]) + f" and {missing_items[-1]}."
        
        # Announce the combined sentence
        speak(alert_text)
        
        # Clear the queue after announcing
        alert_queue.clear()

# Load YOLO model
def load_model():
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5l.pt')
    return model

# Main Streamlit app
def main():
    st.title("PPE Detection System")

    # Initialize session state
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    # Load YOLO model
    model = load_model()

    # Start and Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera", key="start_button"):
            st.session_state.camera_active = True
    with col2:
        if st.button("Stop Camera", key="stop_button"):
            st.session_state.camera_active = False

    # Camera feed placeholder
    frame_placeholder = st.empty()

    # Start video capture if camera is active
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Perform detection
            results = model(frame)

            # Parse results
            detections = results.xyxy[0].numpy()  # Get detections
            detected_items = set()

            # Define upper body PPE items
            upper_body_ppe = {"helmet", "vest", "gloves"}

            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                label = model.names[int(cls)]  # Get class label

                # Only consider upper body PPE items
                if label in upper_body_ppe:
                    detected_items.add(label)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame in Streamlit
            frame_placeholder.image(frame, channels="BGR")

            # Check for missing PPE and add to the alert queue
            missing_ppe = upper_body_ppe - detected_items

            if missing_ppe:
                for item in missing_ppe:
                    if item not in alert_queue:  # Avoid duplicate alerts
                        alert_queue.append(item)

            # Process alerts in the queue
            process_alerts()

            # Add a small delay to avoid overloading the system
            time.sleep(0.1)

        # Release the camera when stopped
        cap.release()
        st.write("Camera stopped.")

# Run the app
if __name__ == "__main__":
    main()