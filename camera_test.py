import cv2

def find_available_cameras(max_cameras_to_check=10):
    available_cameras = []
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available.")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"Camera {i} is not available.")
    return available_cameras

if __name__ == "__main__":
    find_available_cameras()
