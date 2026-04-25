# Generic rule: 'make FS_9.1' compiles FS_9.1.cpp and injects filename into HUD automatically
%: %.cpp
	g++ -std=c++17 -O2 -pthread \
		-DSOURCE_FILENAME='"$<"' \
		$< `pkg-config --cflags --libs opencv4` -o $@

run_fs91:
	DISPLAY=:0 XAUTHORITY=/home/bia/.Xauthority sudo -E ./FS_9.1 2>&1 | tee /tmp/fs_run.log
run:
	g++ -std=c++17 -O2 -pthread rpi5_ar64_10_03_26_v1.cpp `pkg-config --cflags --libs opencv4` -o new_test_arducam64_2
	DISPLAY=:0 ./new_test_arducam64_2

build:
	g++ -std=c++17 -O2 -pthread new_test_arducam64_2.cpp -o new_test_arducam64_2 `pkg-config --cflags --libs opencv4`

sweep:
	g++ -std=c++17 -O2 -pthread new_test_arducam64_2.cpp `pkg-config --cflags --libs opencv4` -o new_test_arducam64_2
	sudo ./new_test_arducam64_2 --sweep

# Targets for the restored version with HTTP remote control threading
build_restored:
	g++ -std=c++17 -O2 -pthread new_test_arducam64_2_restored.cpp -o new_test_arducam64_2_restored `pkg-config --cflags --libs opencv4 gstreamer-1.0`

run_restored: build_restored
	DISPLAY=:0 ./new_test_arducam64_2_restored


# 22_02_26 version
build_22:
	g++ -std=c++17 -O2 -pthread new_test_arducam64_22_02_26.cpp -o new_test_arducam64_22_02_26 $(pkg-config --cflags --libs opencv4)

run_22: build_22
	DISPLAY=:0 ./new_test_arducam64_22_02_26

# rpi5_ar64_10_03_26_v1 version
build_rpi5:
	g++ -std=c++17 -O2 -pthread rpi5_ar64_10_03_26_v1.cpp -o rpi5_ar64_10_03_26_v1 `pkg-config --cflags --libs opencv4`

run_rpi5: build_rpi5
	DISPLAY=:0 ./rpi5_ar64_10_03_26_v1

# ChtGPT5_v3.1
build_v31:
	g++ -std=c++17 -O2 -pthread ChtGPT5_v3.1.cpp `pkg-config --cflags --libs opencv4` -o ChtGPT5_v3.1_bin
run_v31: build_v31
	DISPLAY=:0 ./ChtGPT5_v3.1_bin

# ChtGPT5_prog_rpI5_ar64_v3
build_v3:
	g++ -std=c++17 -O2 -pthread ChtGPT5_prog_rpI5_ar64_v3.cpp `pkg-config --cflags --libs opencv4` -o ChtGPT5_prog_rpI5_ar64_v3

run_v3: build_v3
	DISPLAY=:0 ./ChtGPT5_prog_rpI5_ar64_v3
