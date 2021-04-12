#pragma once

#include <windows.h>
#include "FLibCore.h"

LRESULT CALLBACK window_callback(HWND, UINT, WPARAM, LPARAM);
void init_window(HINSTANCE hInstance, Eigen::VectorXd data1, Eigen::VectorXd data2, Eigen::VectorXd data3, int width, int height);
void display_buffer();
void* get_buffer_memory_ref();
int get_buffer_width();
int get_buffer_height();