import onnxruntime
import numpy as np
import cv2

# Load the ONNX model
session = onnxruntime.InferenceSession("inference_model_2.onnx")
input_name = session.get_inputs()[0].name
GEese_CLASS_ID = 1 
threshold = 0.30  

# Open the video file
video_path = "Download.mp4"  
cap = cv2.VideoCapture(video_path)

# Get video properties (e.g., width, height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare VideoWriter to save the processed video with bounding boxes
output_video_path = "output_video_counted6.mp4
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model (resize and normalize)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize the image to a higher resolution (e.g., 416x416)
    img_resized = cv2.resize(image_rgb, (224, 224))  
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1)) 
    img_input = np.expand_dims(img_input, axis=0)  

    # Run inference
    outputs = session.run(None, {input_name: img_input})
    boxes = outputs[0][0]  
    logits = outputs[1][0]  

    # Get class IDs and scores
    class_ids = np.argmax(logits, axis=-1)
    scores = np.max(logits, axis=-1)

    # Process each detection
    for i in range(len(boxes)):
        if class_ids[i] != GEese_CLASS_ID:
            continue
        if scores[i] < threshold:
            continue

        # Get bounding box coordinates
        cx, cy, bw, bh = boxes[i]
        x1 = int((cx - bw / 2) * w_orig)
        y1 = int((cy - bh / 2) * h_orig)
        x2 = int((cx + bw / 2) * w_orig)
        y2 = int((cy + bh / 2) * h_orig)

        # Draw the bounding box around detected geese
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Display the frame with bounding boxes (for debugging purposes)
    cv2.imshow("Geese Detection", frame)

    # Check detection results for debugging
    print(f"Class IDs: {class_ids}")
    print(f"Scores: {scores}")

    # Break the loop if the 'q' key is pressed (useful for local environments)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
