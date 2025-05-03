import onnxruntime
import numpy as np
import cv2

# Load the ONNX model
session = onnxruntime.InferenceSession("inference_model_2.onnx")
input_name = session.get_inputs()[0].name
GEESE_CLASS_ID = 1 
threshold = 0.25  

# Open the video file
video_path = "Download.mp4"  
cap = cv2.VideoCapture(video_path)

# Get video properties (e.g., width, height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare VideoWriter to save the processed video with bounding boxes
output_video_path = "output_video_counted7.mp4"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Running total of geese detections
total_geese_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model (resize and normalize)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # Resize the image for the model input
    img_resized = cv2.resize(image_rgb, (224, 224))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  
    img_input = np.expand_dims(img_input, axis=0)   

    # Run inference
    outputs = session.run(None, {input_name: img_input})
    boxes = outputs[0][0] 
    logits = outputs[1][0]  

    class_ids = np.argmax(logits, axis=-1)
    scores = np.max(logits, axis=-1)

    # Count and draw detections
    geese_count_frame = 0
    for i in range(len(boxes)):
        if class_ids[i] != GEESE_CLASS_ID or scores[i] < threshold:
            continue

        # Get bounding box coordinates
        cx, cy, bw, bh = boxes[i]
        x1 = int((cx - bw / 2) * w_orig)
        y1 = int((cy - bh / 2) * h_orig)
        x2 = int((cx + bw / 2) * w_orig)
        y2 = int((cy + bh / 2) * h_orig)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (147, 112, 219), 2)  # Lavender box
        geese_count_frame += 1

    total_geese_count += geese_count_frame
    cv2.putText(
        frame,
        f"Geese This Frame: {geese_count_frame}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 105, 180),  # Hot pink
        2
    )

    out.write(frame)
    cv2.imshow("Geese Detection", frame)

    # Optional: press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
