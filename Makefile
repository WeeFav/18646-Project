build:
	nvcc -std=c++17 main.cu model.cpp tgaimage.cpp -o main.x -lstdc++fs

run:
	rm -rf utah_teapot_results
	./main.x utah_teapot

build_baseline:
	cd build/ && cmake --build .

run_baseline:
	rm -rf utah_teapot_results
	./build/tinyrenderer utah_teapot