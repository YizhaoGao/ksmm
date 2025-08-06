// -*- c -*-

#ifndef KERNEL_BS_FIRST_HALF
#define KERNEL_BS_FIRST_HALF

#include "template_kernels_half.cuh"

void best_kernel_bs_first_half(half *input, half *values, half *output, int batch_size, int a, int b, int c, int d, dim3 &blockGrid, dim3 &threadsPerBlock){
	while (1) {
		threadsPerBlock.y = 1;
		if (batch_size == 25088 && a == 32 && b == 64 && c == 64 && d == 1) {
			threadsPerBlock.x = 64;
			blockGrid.x = 32;
			blockGrid.y = 98;
			kernel_bs_first_half4<bool, 64, 32, 256, 16, 16, 4, false, 4><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 128) {
			threadsPerBlock.x = 256;
			blockGrid.x = 1024;
			blockGrid.y = 196;
			kernel_bs_first_half4<bool, 64, 32, 128, 4, 8, 4, false, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 512 && c == 512 && d == 1) {
			threadsPerBlock.x = 256;
			blockGrid.x = 2;
			blockGrid.y = 98;
			kernel_bs_first_half4<bool, 256, 16, 256, 16, 16, 4, false, 16><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
			break;
		}
		if (batch_size == 25088 && a == 1 && b == 128 && c == 512 && d == 128) {
			threadsPerBlock.x = 128;
			blockGrid.x = 128;
			blockGrid.y = 98;
			kernel_bs_first_half4<bool, 128, 32, 256, 16, 16, 4, false, 8><<<blockGrid, threadsPerBlock>>>(input, values, batch_size, output, a, b, c, d);
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
