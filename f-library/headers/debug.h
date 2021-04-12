#pragma once

#include <fstream>

//#define DEBUG_MODE

#define setDebugFile()\
	std::ofstream file;\
	file.open("cout.txt");\
	std::streambuf* sbuf = std::cout.rdbuf();\
	std::cout.rdbuf(file.rdbuf())

#ifdef DEBUG_MODE
#define debug(x)\
	std::cout << (x) << '\n'
#else
#define debug(x)
#endif
