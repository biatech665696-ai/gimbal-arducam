#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

struct Mode { int w; int h; };

static const std::vector<Mode> DEFAULT_MODES = {
    {1920,1080}, {1280,720}, {3840,2160}, {4624,3472}
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --libcamera            Force libcamerasrc pipelines\n"
              << "  --mode=WxH            Request sensor/output mode (e.g. 1920x1080)\n"
              << "  --fps=N               Target framerate (default 30)\n"
              << "  --role=NAME           Stream role: view-finder|video-recording|still-capture (default view-finder)\n"
              << "  --list                List recommended pipelines and exit\n"
              << "  --duration=SECONDS    Auto-exit after duration\n";
}

int main(int argc, char** argv) {
    cv::VideoCapture cap;
    bool force_libcamera = false;
    int req_w = 0, req_h = 0;
    int req_fps = 30;
    int duration_seconds = 0;
    bool list_only = false;
    std::string role = "view-finder"; // default for responsive preview

    for (int i=1;i<argc;++i) {
        std::string arg = argv[i];
        if (arg == "--libcamera") force_libcamera = true;
        else if (arg.rfind("--mode=",0)==0) {
            std::string v = arg.substr(7); auto x=v.find('x');
            if (x!=std::string::npos) { try { req_w=std::stoi(v.substr(0,x)); req_h=std::stoi(v.substr(x+1)); } catch(...){} }
        } else if (arg.rfind("--fps=",0)==0) {
            try { req_fps = std::stoi(arg.substr(6)); } catch(...) {}
        } else if (arg=="--list") list_only = true;
        else if (arg.rfind("--role=",0)==0) {
            std::string v = arg.substr(7);
            if (v=="video-recording" || v=="view-finder" || v=="still-capture") role = v; else {
                std::cerr << "Unknown role '" << v << "' using default view-finder\n";
            }
        }
        else if (arg.rfind("--duration=",0)==0) {
            try { duration_seconds = std::stoi(arg.substr(11)); } catch(...) {}
        } else if (arg=="-h" || arg=="--help") { print_usage(argv[0]); return 0; }
    }

    if (list_only) {
        std::cout << "Recommended libcamerasrc pipelines (modify width/height as supported):\n";
        for (auto &m : DEFAULT_MODES) {
            std::cout << "libcamerasrc stream-role=" << role << " ! video/x-raw,width=" << m.w << ",height=" << m.h << ",framerate=" << req_fps << "/1 ! videoconvert ! videoscale ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2 sync=false" << "\n";
        }
        return 0;
    }

    auto make_libcamera = [&](int w,int h){
        std::ostringstream oss;
        oss << "libcamerasrc stream-role=" << role << " ! video/x-raw,width=" << w << ",height=" << h << ",framerate=" << req_fps << "/1 ! videoconvert ! videoscale ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2 sync=false";
        return oss.str();
    };
    auto make_v4l2 = [&](int w,int h){
        std::ostringstream oss;
        oss << "v4l2src device=/dev/video0 io-mode=2 ! video/x-raw,width=" << w << ",height=" << h << ",format=YUY2,framerate=" << req_fps << "/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2 sync=false";
        return oss.str();
    };

    std::vector<Mode> try_modes;
    if (req_w>0 && req_h>0) try_modes.push_back({req_w,req_h});
    for (auto &m: DEFAULT_MODES) {
        bool already=false; for(auto &t: try_modes){ if(t.w==m.w && t.h==m.h){ already=true; break; } }
        if(!already) try_modes.push_back(m);
    }

    std::string pipeline;
    if (!force_libcamera) {
        // Try direct V4L2 first for requested or 1280x720
        for (auto &m : try_modes) {
            pipeline = make_v4l2(m.w,m.h);
            std::cerr << "Trying V4L2 pipeline: " << pipeline << std::endl;
            if (cap.open(pipeline, cv::CAP_GSTREAMER) && cap.isOpened()) { break; }
        }
    }
    if (!cap.isOpened()) {
        std::cerr << (force_libcamera?"Forced libcamerasrc":"Falling back to libcamerasrc") << " (role=" << role << ")" << std::endl;
        for (auto &m : try_modes) {
            pipeline = make_libcamera(m.w,m.h);
            std::cerr << "Trying libcamera pipeline: " << pipeline << std::endl;
            if (cap.open(pipeline, cv::CAP_GSTREAMER) && cap.isOpened()) { break; }
        }
    }

    if (!cap.isOpened()) {
        std::cerr << "Libcamera primary attempt failed, retrying with alternate role video-recording..." << std::endl;
        std::string alt_role = (role=="video-recording"?"view-finder":"video-recording");
        for (auto &m : try_modes) {
            std::string alt = "libcamerasrc stream-role=" + alt_role + " ! video/x-raw,width=" + std::to_string(m.w) + ",height=" + std::to_string(m.h) + ",framerate=" + std::to_string(req_fps) + "/1 ! videoconvert ! videoscale ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2 sync=false";
            std::cerr << "Trying alternate role pipeline: " << alt << std::endl;
            if (cap.open(alt, cv::CAP_GSTREAMER) && cap.isOpened()) { role = alt_role; break; }
        }
    }
    if (!cap.isOpened()) {
        std::cerr << "All pipelines failed; attempting numeric V4L2 indices as last resort." << std::endl;
        for (int i=0;i<5 && !cap.isOpened();++i) {
            cap.open(i, cv::CAP_V4L2);
            if (cap.isOpened()) {
                std::cout << "Opened camera at index " << i << " (V4L2)." << std::endl;
                break;
            }
        }
    }
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open any camera." << std::endl;
        return -1;
    }

    std::cout << "Press any key in the window to exit. FPS target: " << req_fps << std::endl;

    int got_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int got_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double got_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Negotiated: " << got_w << "x" << got_h << " @ " << got_fps << " (reported) role=" << role << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double last_report = 0.0;

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "ERROR: Failed to grab frame." << std::endl;
            break;
        }

        if (frame.empty()) {
            std::cerr << "ERROR: Grabbed an empty frame." << std::endl;
            continue;
        }

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed - last_report >= 2.0) {
            double fps = frame_count / elapsed;
            std::cout << std::fixed << std::setprecision(2) << "[" << elapsed << "s] FPS=" << fps << " frames=" << frame_count << std::endl;
            last_report = elapsed;
        }
        if (duration_seconds>0 && elapsed >= duration_seconds) {
            std::cout << "Duration reached (" << duration_seconds << "s). Exiting." << std::endl;
            break;
        }

        cv::imshow("Camera Test", frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    // Properly cleanup GStreamer pipeline
    cv::destroyAllWindows();
    if (cap.isOpened()) {
        cap.release();
    }

    return 0;
}
