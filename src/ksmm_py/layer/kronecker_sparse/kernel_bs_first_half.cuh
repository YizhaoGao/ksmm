// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF
#define KERNEL_BS_FIRST_HALF

#include "template_kernels_half.cuh"

void best_kernel_bs_first_half(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 2 && a == 64 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 64 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 32 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 16 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 8 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 4 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 2 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 1 && b == 16 && c == 8 && d == 256) {
			threadsPerBlock.x = 32;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 128 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 64 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 32 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 16 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 8 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 4 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 2 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 1 && b == 16 && c == 8 && d == 128) {
			threadsPerBlock.x = 32;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 64 && b == 16 && c == 8 && d == 1) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 32 && b == 16 && c == 8 && d == 2) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 16 && b == 16 && c == 8 && d == 4) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 8 && b == 16 && c == 8 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 4 && b == 16 && c == 8 && d == 16) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 2 && b == 16 && c == 8 && d == 32) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 1 && b == 16 && c == 8 && d == 64) {
			threadsPerBlock.x = 32;
			blockGrid.x = 64;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 8, 128, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 256 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 128 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 64 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 32 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 16 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 8 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 4 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 2 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 1 && b == 16 && c == 4 && d == 256) {
			threadsPerBlock.x = 16;
			blockGrid.x = 256;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 128 && b == 16 && c == 4 && d == 1) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 64 && b == 16 && c == 4 && d == 2) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 32 && b == 16 && c == 4 && d == 4) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 16 && b == 16 && c == 4 && d == 8) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 8 && b == 16 && c == 4 && d == 16) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 4 && b == 16 && c == 4 && d == 32) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 2 && b == 16 && c == 4 && d == 64) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4096 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 64;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2048 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 32;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 16;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 512 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 8;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 256 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 128 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 2;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 64 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 32 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 16 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 8 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 4 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 2 && a == 1 && b == 16 && c == 4 && d == 128) {
			threadsPerBlock.x = 16;
			blockGrid.x = 128;
			blockGrid.y = 1;
			kernel_bs_first_half4<bool, 16, 4, 64, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 64 && c == 32 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 64, 32, 256, 16, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 32 && c == 32 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 32, 32, 256, 8, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 1024 && a == 32 && b == 16 && c == 32 && d == 2) {
			threadsPerBlock.x = 64;
			blockGrid.x = 64;
			blockGrid.y = 4;
			kernel_bs_first_half4<bool, 16, 16, 256, 4, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 48 && c == 48 && d == 8) {
			threadsPerBlock.x = 32;
			blockGrid.x = 24;
			blockGrid.y = 3136;
			kernel_bs_first_half2<half2, 16, 8, 8, 2, 2, 2, true, 8><<<blockGrid, threadsPerBlock>>>(input, values, 25088, output, a, b, c, d);
			break;
		}
		assert(1 == 0);
		break;
	}
}

#endif
