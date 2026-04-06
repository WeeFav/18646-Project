build_gpu:
	nvcc -std=c++17 main.cu model.cpp tgaimage.cpp -o main.x -lstdc++fs

run_gpu:
	./main.x utah_teapot

build_baseline:
	cd build/ && cmake --build .

run_baseline:
	./build/tinyrenderer utah_teapot