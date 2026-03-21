all:
	g++ $$(pkg-config --cflags opencv4) rpicam-test.cpp -o rpicam-test $$(pkg-config --libs opencv4) 
	g++ $$(pkg-config --cflags opencv4) rpicam-test-without-daemon.cpp -o rpicam-test-without-daemon $$(pkg-config --libs opencv4) 
