/* opencl file with kernels */

#define WIDTH 640
#define HEIGHT 480
#define SQUARE (WIDTH * HEIGHT)
#define DISP_MAX 40
#define DISP_MIN 0
#define DIFF (DISP_MAX - DISP_MIN)

float AD(const uint4 l, const uint4 r) {
	/* AD metric*/

	return (float) ( abs_diff(l.x, r.x) +
				     abs_diff(l.y, r.y) +
		             abs_diff(l.z, r.z) ) / 3;
}

__kernel void kComputeCosts(__global uchar* L,
						    __global uchar* R,
	                        __global float* costs) {
/*  kernel to compute raw-costs with AD-Census metric only wothout 
    aggregation and improvements  */

	const int3 xyz = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
	// get the current pixel colour  from left and right images 
	const uint lCoord = 3 * (xyz.x + xyz.y * WIDTH);
	const uint4 pixL = { L[lCoord], 
						 L[lCoord + 1],
						 L[lCoord + 2],
						 0 };

	const uint rCoord = 3 * (xyz.x + xyz.y * WIDTH - xyz.z - DISP_MIN);
	const uint4 pixR = { R[rCoord],
						 R[rCoord + 1],
						 R[rCoord + 2],
					     0 };

	costs[xyz.x + xyz.y * WIDTH + xyz.z * SQUARE] = AD(pixL, pixR);
}

__kernel void kGetDisparityMap(__global float* costs,
							   __global float* disp) {
/*  find minimal costs values and construct isparity map */

	const int2 xy = (int2)(get_global_id(0), get_global_id(1));

	float min = costs[xy.x + xy.y * WIDTH];
	float minInd = 0;

	// go through z-axis of (x,y) position
	for (ushort z = 1; z < DIFF; z++) {
		const float value = costs[xy.x + xy.y * WIDTH + z * SQUARE];

		if (value < min) { min = value; minInd = z; }
	}

	disp[xy.x + xy.y * WIDTH] = minInd;
}