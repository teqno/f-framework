#include <stdlib.h>
#include <tchar.h>
#include <functional>
#include "window.h"

static TCHAR szWindowClass[] = _T("FLib Demo");
static TCHAR szTitle[] = _T("Windows Desktop Guided Tour Application");

HDC hdc;
bool isRunning = true;
int buffer_width;
int buffer_height;
void* buffer_memory;
BITMAPINFO buffer_bitmap_info;

int state = 0;

void clearBuffer() {
	void* buffer_memory = get_buffer_memory_ref();
	int buffer_height = get_buffer_height();
	int buffer_width = get_buffer_width();

	unsigned int* pixel = (unsigned int*)buffer_memory;
	for (int y = 0; y < buffer_height; y++) {
		for (int x = 0; x < buffer_width; x++) {
			*pixel = 0xffffff;
			pixel++;
		}
	}

	display_buffer();
}

void action(Eigen::VectorXd data1, Eigen::VectorXd data2) {
	void* buffer_memory = get_buffer_memory_ref();
	int buffer_height = get_buffer_height();
	int buffer_width = get_buffer_width();
	double coef_x = (double)data1.size() / buffer_width;
	double coef_y = (data1.maxCoeff() - data1.minCoeff()) / buffer_height;

	Eigen::VectorXd biased_y = data1.array() - data1.minCoeff();

	unsigned int* pixel = (unsigned int*)buffer_memory;
	for (int y = 0; y < buffer_height; y++) {
		for (int x = 0; x < buffer_width; x++) {
			int coeffed_x = coef_x * x;

			if (coeffed_x < biased_y.size() && y - (int)(biased_y(coeffed_x) / coef_y + 2) < 0 && y - (int)(biased_y(coeffed_x) / coef_y - 2) > 0) {
				*pixel = 0xff0000;
			}
			else {
				*pixel = 0xffffff;
			}
			pixel++;
		}
	}

	biased_y = data2.array() - data1.minCoeff(); // biased relative to data1

	pixel = (unsigned int*)buffer_memory;
	for (int y = 0; y < buffer_height; y++) {
		for (int x = 0; x < buffer_width; x++) {
			int coeffed_x = coef_x * x;

			if (coeffed_x < biased_y.size() && y - (int)(biased_y(coeffed_x) / coef_y + 2) < 0 && y - (int)(biased_y(coeffed_x) / coef_y - 2) > 0) {
				*pixel = 0x00ff00;
			}
			pixel++;
		}
	}

	display_buffer();
}

void displayState(Eigen::VectorXd data1, Eigen::VectorXd data2, Eigen::VectorXd data3) {
	switch (state)
	{
		case 0: {
			action(data1, data2);
			break;
		}
		case 1: {
			action(data3, Eigen::VectorXd());
			break;
		}
		default:
			break;
	}
}

void init_window(HINSTANCE hInstance, Eigen::VectorXd data1, Eigen::VectorXd data2, Eigen::VectorXd data3, int width, int height) {
	WNDCLASS window_class = {};
	window_class.style = CS_HREDRAW | CS_VREDRAW;
	window_class.lpszClassName = szWindowClass;
	window_class.lpfnWndProc = window_callback;

	RegisterClass(&window_class);

	HWND hwnd =
		CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
			CW_USEDEFAULT, width, height, NULL, NULL, hInstance, NULL);
	hdc = GetDC(hwnd);

	while (isRunning) {
		MSG msg;
		while (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		displayState(data1, data2, data3);
	}
}

LRESULT CALLBACK window_callback(HWND hwnd, UINT message, WPARAM wParam,
	LPARAM lParam) {
	TCHAR greeting[] = _T("Hello, Windows desktop!");

	switch (message) {
		case WM_KEYDOWN: {
			switch (wParam)
			{
				case VK_RETURN: {
					if (state == 0) {
						state = 1;
					}
					else {
						state = 0;
					}
					break;
				}
				case VK_ESCAPE: {
					isRunning = false;
					break;
				}
				default:
					break;
			}
			break;
		}
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