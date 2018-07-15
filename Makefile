all:
	g++ -std=c++11 -Wall -Wextra -pedantic -O3 -I. main.cpp -lglfw -o rasterizer
