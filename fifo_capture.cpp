#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

/*
 * fifo_capture.cpp
 * Simple MJPEG frame reader from a named FIFO produced by libcamera-vid.
 * Usage:
 *   ./fifo_capture --fifo /tmp/cam.mjpg --spawn "libcamera-vid --codec mjpeg --width 1920 --height 1080 --framerate 30 --timeout 0 --nopreview -o /tmp/cam.mjpg" --duration 10
 *   ./fifo_capture --fifo /tmp/cam.mjpg --width 1920 --height 1080 --save-jpeg-dir frames
 * Steps:
 *   1. Ensure libcamera-vid installed.
 *   2. Program will mkfifo if missing, optionally spawn producer command.
 *   3. Reads byte stream, extracts JPEG frames (SOI FF D8 ... EOI FF D9), decodes via cv::imdecode.
 *   4. Shows frames and computes FPS.
 */

static std::atomic<bool> keep_running{true};

void signal_handler(int){ keep_running = false; }

struct Options {
    std::string fifo = "/tmp/cam.mjpg";
    std::string spawn_cmd; // optional libcamera-vid invocation
    int width = 0; // advisory only
    int height = 0; // advisory only
    int duration = 0; // seconds
    std::string save_dir; // optional directory to save JPEGs
    bool verbose = false;
};

void print_usage(const char* prog){
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --fifo PATH           FIFO path (default /tmp/cam.mjpg)\n"
              << "  --spawn CMD           Spawn libcamera-vid or other producer command\n"
              << "  --width W            Advisory display width\n"
              << "  --height H           Advisory display height\n"
              << "  --duration S         Auto-exit after S seconds\n"
              << "  --save-jpeg-dir DIR  Save decoded JPEG frames to DIR\n"
              << "  --verbose            Extra logging\n"
              << "  -h,--help            Show help\n";
}

bool ensure_fifo(const std::string& path){
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISFIFO(st.st_mode)) return true;
        std::cerr << "Path exists but is not FIFO: " << path << std::endl;
        return false;
    }
    if (mkfifo(path.c_str(), 0666) != 0) {
        perror("mkfifo");
        return false;
    }
    return true;
}

int main(int argc, char** argv){
    signal(SIGINT, signal_handler);
    Options opt;
    for (int i=1;i<argc;++i){
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--verbose") opt.verbose = true;
        else if (a.rfind("--fifo",0)==0){
            if (a=="--fifo" && i+1<argc) opt.fifo = argv[++i];
            else if (a.find('=')!=std::string::npos) opt.fifo = a.substr(a.find('=')+1);
        } else if (a.rfind("--spawn",0)==0){
            if (a=="--spawn" && i+1<argc) opt.spawn_cmd = argv[++i];
            else if (a.find('=')!=std::string::npos) opt.spawn_cmd = a.substr(a.find('=')+1);
        } else if (a.rfind("--width",0)==0){
            if (a=="--width" && i+1<argc) opt.width = std::stoi(argv[++i]);
            else if (a.find('=')!=std::string::npos) opt.width = std::stoi(a.substr(a.find('=')+1));
        } else if (a.rfind("--height",0)==0){
            if (a=="--height" && i+1<argc) opt.height = std::stoi(argv[++i]);
            else if (a.find('=')!=std::string::npos) opt.height = std::stoi(a.substr(a.find('=')+1));
        } else if (a.rfind("--duration",0)==0){
            if (a=="--duration" && i+1<argc) opt.duration = std::stoi(argv[++i]);
            else if (a.find('=')!=std::string::npos) opt.duration = std::stoi(a.substr(a.find('=')+1));
        } else if (a.rfind("--save-jpeg-dir",0)==0){
            if (a=="--save-jpeg-dir" && i+1<argc) opt.save_dir = argv[++i];
            else if (a.find('=')!=std::string::npos) opt.save_dir = a.substr(a.find('=')+1);
        } else {
            std::cerr << "Unknown argument: " << a << std::endl; return 1;
        }
    }

    if (!opt.save_dir.empty()) {
        struct stat st{}; if (stat(opt.save_dir.c_str(), &st)!=0){
            if (mkdir(opt.save_dir.c_str(), 0755)!=0){ perror("mkdir save_dir"); return 1; }
        }
    }

    if (!ensure_fifo(opt.fifo)) return 1;

    if (!opt.spawn_cmd.empty()) {
        // Spawn producer asynchronously using system() & ampersand to detach.
        std::string cmd = opt.spawn_cmd + " &";
        std::cerr << "Spawning producer: " << opt.spawn_cmd << std::endl;
        int rc = system(cmd.c_str());
        if (rc == -1) std::cerr << "Warning: failed to spawn command" << std::endl;
        // Give producer some time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    int fd = open(opt.fifo.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) { perror("open fifo"); return 1; }

    std::vector<unsigned char> buffer; buffer.reserve(2*1024*1024);
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;
    int frame_count = 0;
    int saved_count = 0;

    std::cout << "Reading MJPEG from FIFO: " << opt.fifo << std::endl;

    while (keep_running) {
        unsigned char temp[8192];
        ssize_t r = read(fd, temp, sizeof(temp));
        if (r > 0) {
            buffer.insert(buffer.end(), temp, temp + r);
            // Search for JPEG frames: SOI 0xFFD8 ... EOI 0xFFD9
            size_t pos = 0;
            while (true) {
                // find SOI
                while (pos + 1 < buffer.size() && !(buffer[pos]==0xFF && buffer[pos+1]==0xD8)) pos++;
                if (pos + 1 >= buffer.size()) break;
                size_t soi = pos;
                // find EOI after SOI
                size_t eoi = soi + 2;
                bool found_eoi = false;
                while (eoi + 1 < buffer.size()) {
                    if (buffer[eoi]==0xFF && buffer[eoi+1]==0xD9) { found_eoi = true; eoi += 2; break; }
                    eoi++;
                }
                if (!found_eoi) break; // need more data
                // Extract frame
                std::vector<unsigned char> jpeg(buffer.begin()+soi, buffer.begin()+eoi);
                // Erase from buffer up to eoi
                buffer.erase(buffer.begin(), buffer.begin()+eoi);
                pos = 0; // reset search

                cv::Mat raw(1, (int)jpeg.size(), CV_8UC1, jpeg.data());
                cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
                if (!img.empty()) {
                    frame_count++;
                    if (opt.width>0 && opt.height>0) {
                        cv::resize(img, img, cv::Size(opt.width, opt.height));
                    }
                    cv::imshow("FIFO Capture", img);
                    int key = cv::waitKey(1);
                    if (key >= 0) { keep_running = false; break; }
                    if (!opt.save_dir.empty()) {
                        std::string fname = opt.save_dir + "/frame_" + std::to_string(saved_count++) + ".jpg";
                        cv::imwrite(fname, img);
                    }
                } else if (opt.verbose) {
                    std::cerr << "Failed to decode JPEG frame (size=" << jpeg.size() << ")" << std::endl;
                }
            }
        } else {
            // No data currently; sleep briefly
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        if (elapsed - std::chrono::duration<double>(t_last_report - t_start).count() >= 2.0) {
            double fps = frame_count / elapsed;
            std::cout << std::fixed << std::setprecision(2) << "[" << elapsed << "s] Frames=" << frame_count << " FPS=" << fps << std::endl;
            t_last_report = now;
        }
        if (opt.duration>0 && elapsed >= opt.duration) {
            std::cout << "Duration reached. Exiting." << std::endl;
            break;
        }
    }

    close(fd);
    cv::destroyAllWindows();
    std::cout << "Total frames: " << frame_count << std::endl;
    return 0;
}
