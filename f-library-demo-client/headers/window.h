#pragma once

#include <windows.h>

LRESULT CALLBACK window_callback(HWND, UINT, WPARAM, LPARAM);
void init_window(HINSTANCE hInstance, void (*action)());
void display_buffer();
void* get_buffer_memory_ref();
int get_buffer_width();
int get_buffer_height();