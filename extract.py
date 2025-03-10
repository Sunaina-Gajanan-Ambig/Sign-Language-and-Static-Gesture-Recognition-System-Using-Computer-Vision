import cv2
import numpy as np
from keras.models import model_from_json

# Model loading
json_file = open("signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x48.h5")

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Shape should match model input
    feature = feature.astype("float32")
    return feature / 255.0  # Normalize to [0,1]

# Set up webcam
cap = cv2.VideoCapture(0)
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

confidence_threshold = 50  # You can adjust this threshold for blank gesture

while True:
    _, frame = cap.read()
    if frame is None:
        print("Error: No frame captured.")
        break
    
    # Define ROI for hand gesture
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]

    # Convert to grayscale for better blank gesture detection
    cropframe_gray = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)

    # Check for blank frame based on average pixel intensity (simple thresholding)
    avg_intensity = np.mean(cropframe_gray)
    print(f"Avg Intensity: {avg_intensity}")  # For debugging purpose
    
    if avg_intensity < 50:  # Adjust this threshold value based on testing
        # If the average intensity is below a certain value, it's likely blank
        prediction_label = "blank"
        confidence = 100.0
    else:
        cropframe_resized = cv2.resize(cropframe_gray, (48, 48))  # Resize for model
        cropframe_resized = extract_features(cropframe_resized)
        
        # Model prediction
        pred = model.predict(cropframe_resized)
        confidence = np.max(pred) * 100

        # If confidence is below the threshold, treat it as blank
        if confidence < confidence_threshold:
            prediction_label = "blank"
        else:
            prediction_label = label[pred.argmax()]

    # Display prediction and confidence on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    accu = "{:.2f}".format(confidence)
    cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show output
    cv2.imshow("output", frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(27) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
