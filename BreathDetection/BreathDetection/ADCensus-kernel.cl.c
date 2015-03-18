/* opencl file with kernels */

#define WIDTH 640
#define HEIGHT 480
#define SQUARE (WIDTH * HEIGHT)
#define DISP_MAX 100
#define DISP_MIN 0

__kernel void kComputeCosts(__global uchar* L,
						    __global uchar* R,
	                        __global float* costs) {
/*  kernel to compute raw-costs with AD-Census metric only wothout 
    aggregation and improvements  */

	const int3 xyz = (get_global_id(0), get_global_id(1), get_global_id(2));
	
	// get the current pixel colour  from left and right images 
	const uint4 pixL = { L[ 3 * (xyz.x + xyz.x * WIDTH) ], 
					     L[ 3 * (xyz.x + xyz.x * WIDTH) + 1 ], 
						 L[ 3 * (xyz.x + xyz.x * WIDTH) + 2 ], 
						 0 };

	const uint4 pixR = { R[3 * (xyz.x + xyz.x * WIDTH - xyz.z - DISP_MIN)],
						 R[3 * (xyz.x + xyz.x * WIDTH - xyz.z - DISP_MIN) + 1],
						 R[3 * (xyz.x + xyz.x * WIDTH - xyz.z - DISP_MIN) + 2],
					     0 };

	costs[xyz.x + xyz.y * WIDTH + xyz.z * SQUARE] = (float)(abs_diff(pixL.x, pixR.x) +
															abs_diff(pixL.y, pixR.y) +
															abs_diff(pixL.z, pixR.z)
															) / 3;
}