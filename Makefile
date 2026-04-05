all:
	nvcc -std=c++17 main.cu model.cpp -o main.x -lstdc++fs