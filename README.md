# new_test_arducam64

C++ OpenCV motion detection and tracking sample for Raspberry Pi (Arducam 64MP via libcamera + GStreamer), with low-latency capture and simple ID tracking.

## Build

Requirements: OpenCV 4 development packages installed (`libopencv-dev`), GStreamer/libcamera available.

```bash
cd /home/bia/projects
g++ -O3 -std=c++17 new_test_arducam64.cpp $(pkg-config --cflags --libs opencv4) -o new_test_arducam64
```

## Run

If PipeWire/WirePlumber auto-grab the camera, stop them first:
```bash
systemctl --user stop pipewire.service pipewire.socket pipewire-pulse.service pipewire-pulse.socket wireplumber.service wireplumber.socket
```
Run the program:
```bash
./new_test_arducam64
```
Press `q` or `ESC` to exit.

## Notes
- Uses a GStreamer pipeline (libcamerasrc → NV12 → videoconvert → BGR → appsink) for low latency.
- Motion via MOG2 background subtraction, contour filtering, smallest-object emphasis, and simple nearest-neighbor ID tracking.
