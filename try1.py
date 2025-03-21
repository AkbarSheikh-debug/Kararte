import cv2
import cvzone
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time  # Import the time module for cooldown tracking

# Load the YOLO models
fighter_detection_model = YOLO('v11_fighter_detection_karate.pt')  # Model to detect fighters
pose_estimation_model = YOLO('yolov8m-pose.pt')  # Model to estimate pose

# Initialize variables
hand_up_thresh = 150
hand_down_thresh = 80
leg_up_thresh = 110
leg_down_thresh = 170

# Cooldown period in seconds (adjust as needed)
cooldown_time = 1.0  # 1 second cooldown

# Initialize video capture
cap = cv2.VideoCapture("input2.mp4")  # Use video file (change to 0 for webcam)


# Function to calculate the angle between three points
def angle(px1, py1, px2, py2, px3, py3):
    v1 = np.array([px1, py1]) - np.array([px2, py2])
    v2 = np.array([px3, py3]) - np.array([px2, py2])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# Function to interpolate player positions
def interpolate_player_positions(player_positions):
    # Extract player IDs and their positions
    player_ids = set()
    for frame_positions in player_positions:
        for player_dict in frame_positions:
            player_ids.update(player_dict.keys())

    # Create a dictionary to store positions for each player
    player_data = {player_id: [] for player_id in player_ids}

    # Populate the dictionary with positions
    for frame_positions in player_positions:
        for player_dict in frame_positions:
            for player_id, position in player_dict.items():
                player_data[player_id].append(position)

    # Interpolate missing positions for each player
    for player_id in player_data:
        df = pd.DataFrame(player_data[player_id], columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        player_data[player_id] = df.to_numpy().tolist()

    # Reconstruct the player positions list
    interpolated_positions = []
    for i in range(len(player_positions)):
        frame_positions = []
        for player_id in player_data:
            if i < len(player_data[player_id]):
                frame_positions.append({player_id: player_data[player_id][i]})
        interpolated_positions.append(frame_positions)

    return interpolated_positions


# Dictionary to store angles, positions, and counters for each player
player_angles = {}
player_positions_history = []
player_counters = {}  # To store punch and kick counters for each player
player_cooldowns = {}  # To track cooldown timers for each player

# Main loop for video capture and processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))

    # Detect and track fighters using the fighter detection model
    fighter_results = fighter_detection_model.track(frame, persist=True)  # Enable tracking

    if fighter_results[0].boxes.id is not None:  # Check if tracking IDs are available
        boxes = fighter_results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        ids = fighter_results[0].boxes.id.cpu().numpy().astype(int)  # Get tracking IDs
        class_ids = fighter_results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs
        class_names = fighter_results[0].names  # Get class names

        current_player_positions = []
        for box, id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            current_player_positions.append({id: [x1, y1, x2, y2]})

            # Get the original label from the model
            player_label = class_names[class_id]  # e.g., "black_fighter" or "white_fighter"

            # Initialize counters and cooldowns for the player if not already done
            if id not in player_counters:
                player_counters[id] = {
                    "left_punch_counter": 0,
                    "right_punch_counter": 0,
                    "left_kick_counter": 0,
                    "right_kick_counter": 0,
                    "punch_left": False,
                    "punch_right": False,
                    "kick_left": False,
                    "kick_right": False
                }
                player_cooldowns[id] = {
                    "left_punch_last_time": 0,
                    "right_punch_last_time": 0,
                    "left_kick_last_time": 0,
                    "right_kick_last_time": 0
                }

            # Crop the region of interest (ROI) for the detected fighter
            fighter_roi = frame[int(y1):int(y2), int(x1):int(x2)]

            # Estimate pose for the detected fighter using the pose estimation model
            pose_results = pose_estimation_model(fighter_roi)
            for pose in pose_results:
                keypoints = pose.keypoints.xy.cpu().numpy()  # Get keypoints for the fighter
                for person_keypoints in keypoints:  # Iterate through each person (if multiple)
                    if len(person_keypoints) > 16:  # Ensure all keypoints are available
                        # Keypoints for the arms
                        cx1, cy1 = int(person_keypoints[5][0]), int(person_keypoints[5][1])  # Left shoulder
                        cx2, cy2 = int(person_keypoints[7][0]), int(person_keypoints[7][1])  # Left elbow
                        cx3, cy3 = int(person_keypoints[9][0]), int(person_keypoints[9][1])  # Left wrist
                        cx4, cy4 = int(person_keypoints[6][0]), int(person_keypoints[6][1])  # Right shoulder
                        cx5, cy5 = int(person_keypoints[8][0]), int(person_keypoints[8][1])  # Right elbow
                        cx6, cy6 = int(person_keypoints[10][0]), int(person_keypoints[10][1])  # Right wrist

                        # Keypoints for the legs
                        cx7, cy7 = int(person_keypoints[11][0]), int(person_keypoints[11][1])  # Left hip
                        cx8, cy8 = int(person_keypoints[13][0]), int(person_keypoints[13][1])  # Left knee
                        cx9, cy9 = int(person_keypoints[15][0]), int(person_keypoints[15][1])  # Left ankle
                        cx10, cy10 = int(person_keypoints[12][0]), int(person_keypoints[12][1])  # Right hip
                        cx11, cy11 = int(person_keypoints[14][0]), int(person_keypoints[14][1])  # Right knee
                        cx12, cy12 = int(person_keypoints[16][0]), int(person_keypoints[16][1])  # Right ankle

                        # Calculate angles
                        left_hand_angle = angle(cx1, cy1, cx2, cy2, cx3, cy3)  # Left arm angle
                        right_hand_angle = angle(cx4, cy4, cx5, cy5, cx6, cy6)  # Right arm angle
                        left_leg_angle = angle(cx7, cy7, cx8, cy8, cx9, cy9)  # Left leg angle
                        right_leg_angle = angle(cx10, cy10, cx11, cy11, cx12, cy12)  # Right leg angle

                        # Store angles for the player
                        player_angles[id] = {
                            "player_label": player_label,  # Store player label
                            "left_hand_angle": left_hand_angle,
                            "right_hand_angle": right_hand_angle,
                            "left_leg_angle": left_leg_angle,
                            "right_leg_angle": right_leg_angle
                        }

                        # Get the current time
                        current_time = time.time()

                        # Punch and kick detection logic with cooldown
                        # Left Punch
                        if (left_hand_angle >= hand_down_thresh and not player_counters[id]["punch_left"] and
                                (current_time - player_cooldowns[id]["left_punch_last_time"]) >= cooldown_time):
                            player_counters[id]["punch_left"] = True
                            player_cooldowns[id]["left_punch_last_time"] = current_time
                        elif (left_hand_angle >= hand_up_thresh and player_counters[id]["punch_left"] and
                              (current_time - player_cooldowns[id]["left_punch_last_time"]) >= cooldown_time):
                            player_counters[id]["left_punch_counter"] += 1
                            player_counters[id]["punch_left"] = False

                        # Right Punch
                        if (right_hand_angle >= hand_down_thresh and not player_counters[id]["punch_right"] and
                                (current_time - player_cooldowns[id]["right_punch_last_time"]) >= cooldown_time):
                            player_counters[id]["punch_right"] = True
                            player_cooldowns[id]["right_punch_last_time"] = current_time
                        elif (right_hand_angle >= hand_up_thresh and player_counters[id]["punch_right"] and
                              (current_time - player_cooldowns[id]["right_punch_last_time"]) >= cooldown_time):
                            player_counters[id]["right_punch_counter"] += 1
                            player_counters[id]["punch_right"] = False

                        # Left Kick
                        if (left_leg_angle >= leg_down_thresh and not player_counters[id]["kick_left"] and
                                (current_time - player_cooldowns[id]["left_kick_last_time"]) >= cooldown_time):
                            player_counters[id]["kick_left"] = True
                            player_cooldowns[id]["left_kick_last_time"] = current_time
                        elif (left_leg_angle >= leg_up_thresh and player_counters[id]["kick_left"] and
                              (current_time - player_cooldowns[id]["left_kick_last_time"]) >= cooldown_time):
                            player_counters[id]["left_kick_counter"] += 1
                            player_counters[id]["kick_left"] = False

                        # Right Kick
                        if (right_leg_angle >= leg_down_thresh and not player_counters[id]["kick_right"] and
                                (current_time - player_cooldowns[id]["right_kick_last_time"]) >= cooldown_time):
                            player_counters[id]["kick_right"] = True
                            player_cooldowns[id]["right_kick_last_time"] = current_time
                        elif (right_leg_angle >= leg_up_thresh and player_counters[id]["kick_right"] and
                              (current_time - player_cooldowns[id]["right_kick_last_time"]) >= cooldown_time):
                            player_counters[id]["right_kick_counter"] += 1
                            player_counters[id]["kick_right"] = False

                        # Draw keypoints on the original frame (adjust coordinates to global frame)
                        cv2.circle(frame, (int(x1) + cx1, int(y1) + cy1), 4, (255, 0, 0), -1)  # Left shoulder
                        cv2.circle(frame, (int(x1) + cx2, int(y1) + cy2), 4, (255, 0, 0), -1)  # Left elbow
                        cv2.circle(frame, (int(x1) + cx3, int(y1) + cy3), 4, (255, 0, 0), -1)  # Left wrist
                        cv2.circle(frame, (int(x1) + cx4, int(y1) + cy4), 4, (0, 255, 0), -1)  # Right shoulder
                        cv2.circle(frame, (int(x1) + cx5, int(y1) + cy5), 4, (0, 255, 0), -1)  # Right elbow
                        cv2.circle(frame, (int(x1) + cx6, int(y1) + cy6), 4, (0, 255, 0), -1)  # Right wrist
                        cv2.circle(frame, (int(x1) + cx7, int(y1) + cy7), 4, (0, 0, 255), -1)  # Left hip
                        cv2.circle(frame, (int(x1) + cx8, int(y1) + cy8), 4, (0, 0, 255), -1)  # Left knee
                        cv2.circle(frame, (int(x1) + cx9, int(y1) + cy9), 4, (0, 0, 255), -1)  # Left ankle
                        cv2.circle(frame, (int(x1) + cx10, int(y1) + cy10), 4, (255, 255, 0), -1)  # Right hip
                        cv2.circle(frame, (int(x1) + cx11, int(y1) + cy11), 4, (255, 255, 0), -1)  # Right knee
                        cv2.circle(frame, (int(x1) + cx12, int(y1) + cy12), 4, (255, 255, 0), -1)  # Right ankle

            # Draw bounding box around the detected fighter
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add player label above the bounding box
            cv2.putText(frame, player_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Store current player positions
        player_positions_history.append(current_player_positions)

        # Interpolate player positions to reduce flickering
        if len(player_positions_history) > 1:
            interpolated_positions = interpolate_player_positions(player_positions_history)
            for i, frame_positions in enumerate(interpolated_positions):
                if i < len(boxes):
                    for pos in frame_positions:
                        for player_id, position in pos.items():
                            x1, y1, x2, y2 = position  # Correctly unpack the position
                            boxes[i] = [x1, y1, x2, y2]

        # Display counters for each player
        # y_offset = 50
        for player_id, counters in player_counters.items():
            player_label = player_angles[player_id]["player_label"]  # Get player label

            # Check if the player is the white player
            if player_label == "white_fighter":
                # Display white player's counters on the right-down side of the screen
                cvzone.putTextRect(frame, f'{player_label} Left Punch: {counters["left_punch_counter"]}', (50, 50),
                                   1, 1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Right Punch: {counters["right_punch_counter"]}', (50, 100),
                                   1, 1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Left Kick: {counters["left_kick_counter"]}', (50, 150), 1,
                                   1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Right Kick: {counters["right_kick_counter"]}', (50, 200),
                                   1, 1)
                # y_offset += 60  # Add extra space between players
            else:
                # Display other player's counters on the left side of the screen
                cvzone.putTextRect(frame, f'{player_label} Left Punch: {counters["left_punch_counter"]}', (750, 300), 1,
                                   1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Right Punch: {counters["right_punch_counter"]}', (750, 350),
                                   1, 1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Left Kick: {counters["left_kick_counter"]}', (750, 400), 1,
                                   1)
                # y_offset += 80
                cvzone.putTextRect(frame, f'{player_label} Right Kick: {counters["right_kick_counter"]}', (750, 450), 1,
                                   1)
                # y_offset += 60  # Add extra space between players

    # Display the frame
    cv2.imshow("Fighter Pose Estimation", frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()