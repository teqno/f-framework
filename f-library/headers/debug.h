#pragma once

#include <fstream>

#define setDebugFile()\
	std::ofstream file;\
	file.open("cout.txt");\
	std::streambuf* sbuf = std::cout.rdbuf();\
	std::cout.rdbuf(file.rdbuf())

#define debug(x)\
	std::cout << (x) << '\n'