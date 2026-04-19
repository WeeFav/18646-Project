TILE_WIDTH ?= 16
TILE_HEIGHT ?= 16
GPU_BIN := main_$(TILE_WIDTH)x$(TILE_HEIGHT).x

build_gpu:
	nvcc -std=c++17 -DTILE_WIDTH=$(TILE_WIDTH) -DTILE_HEIGHT=$(TILE_HEIGHT) main.cu model.cpp tgaimage.cpp -o $(GPU_BIN) -lstdc++fs

run_gpu:
	./$(GPU_BIN) utah_teapot

build_cpu:
	cd build/ && cmake --build .

run_cpu:
	./build/tinyrenderer utah_teapot