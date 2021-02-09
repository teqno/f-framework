#include <stdlib.h>
#include <tchar.h>
#include "window.h"

static TCHAR szWindowClass[] = _T("FLib Demo");
static TCHAR szTitle[] = _T("Windows Desktop Guided Tour Application");

HDC hdc;
bool isRunning = true;
int buffer_width;
int buffer_height;
void* buffer_memory;
BITMAPINFO buffer_bitmap_info;

void init_window(HINSTANCE hInstance, void (*action)()) {
	WNDCLASS window_class = {};
	window_class.style = CS_HREDRAW | CS_VREDRAW;
	window_class.lpszClassName = szWindowClass;
	window_class.lpfnWndProc = window_callback;

	RegisterClass(&window_class);

	HWND hwnd =
		CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
			CW_USEDEFAULT, 1000, 700, NULL, NULL, hInstance, NULL);
	hdc = GetDC(hwnd);

	while (isRunning) {
		MSG msg;
		while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		action();
		Sleep(16.666);
	}
}

LRESULT CALLBACK window_callback(HWND hwnd, UINT message, WPARAM wParam,
	LPARAM lParam) {
	TCHAR greeting[] = _T("Hello, Windows desktop!");

	switch (message) {
		case WM_SIZE: {
			RECT rect;
			GetClientRect(hwnd, &rect);
			buffer_width = rect.right - rect.left;
			buffer_height = rect.bottom - rect.top;

			int buffer_size = buffer_width * buffer_height * sizeof(unsigned int);

			if (buffer_memory) {
				VirtualFree(buffer_memory, 0, MEM_RELEASE);
			}

			buffer_memory =
				VirtualAlloc(0, buffer_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

			buffer_bitmap_info.bmiHeader.biSize = sizeof(buffer_bitmap_info.bmiHeader);
			buffer_bitmap_info.bmiHeader.biWidth = buffer_width;
			buffer_bitmap_info.bmiHeader.biHeight = buffer_height;
			buffer_bitmap_info.bmiHeader.biPlanes = 1;
			buffer_bitmap_info.bmiHeader.biBitCount = 32;
			buffer_bitmap_info.bmiHeader.biCompression = BI_RGB;
			break;
		}
		case WM_CLOSE:
		case WM_DESTROY: {
			isRunning = false;
			break;
		}
		default: {
			return DefWindowProc(hwnd, message, wParam, lParam);
			break;
		}
	}

	return 0;
}

void display_buffer() {
	StretchDIBits(hdc, 0, 0, buffer_width, buffer_height, 0, 0, buffer_width, buffer_height, buffer_memory, &buffer_bitmap_info, DIB_RGB_COLORS, SRCCOPY);
}

void* get_buffer_memory_ref() {
	return buffer_memory;
}

int get_buffer_width() {
	return buffer_width;
}

int get_buffer_height() {
	return buffer_height;
}