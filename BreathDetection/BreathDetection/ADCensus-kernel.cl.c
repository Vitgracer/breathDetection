/* opencl file with kernels */

#define WIDTH 640
#define HEIGHT 480
#define SQUARE (WIDTH * HEIGHT)
#define DISP_MAX 40
#define DISP_MIN 0
#define DIFF (DISP_MAX - DISP_MIN)
#define AD_LAMBD 10
#define CENSUS_LAMBD 30
#define TAU1 20 
#define L1 34

float AD(const uint4 l, const uint4 r) {
	/* AD metric*/

	return (float) ( abs_diff(l.x, r.x) +
				     abs_diff(l.y, r.y) +
		             abs_diff(l.z, r.z) ) / 3;
}

float Census(const int2 l, const int2 r, __global uchar* lImg, __global uchar* rImg, char channel) {
	/* Census metric */

	const uchar lCenter = lImg[ 3 * (l.x + l.y * WIDTH) + channel];
	const uchar rCenter = rImg[ 3 * (r.x + r.y * WIDTH) + channel];
	
	float hamming = 0.0;

	for (char y = -3; y < 4; y++) {
		for (char x = -4; x < 5; x++) {
			
			// if center of the box - skip
			if (!(x || y)) continue;

			const uchar lVal = lImg[3 * (l.x + x + (l.y + y) * WIDTH) + channel];
			const uchar rVal = rImg[3 * (r.x + x + (r.y + y) * WIDTH) + channel];

			if ((lVal - lCenter) * (rVal - rCenter) < 0) hamming += 1.0;
		}
	}

	return hamming;
}

float ADCensus(float AD, float Census) {
	/* ADCensus metric */
	
	return (1 - exp(-1 * AD / AD_LAMBD) + 1 - exp(-1 * Census / CENSUS_LAMBD) );
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
	
	// get AD and Census result 
	const float resAD = AD(pixL, pixR);
	
	const int2 censusPointL = (int2)(xyz.x, xyz.y);
	const int2 censusPointR = (int2)(xyz.x - xyz.z - DISP_MIN, xyz.y);

	const float resCensus = Census(censusPointL, censusPointR, L, R, 0) +
							Census(censusPointL, censusPointR, L, R, 1) +
							Census(censusPointL, censusPointR, L, R, 2);

	costs[xyz.x + xyz.y * WIDTH + xyz.z * SQUARE] = ADCensus(resAD, resCensus);
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

float DC(__global uchar* img, const int2 p1, const int2 p2) {
/* DC metric */
	const int coord1 = 3 * (p1.x + p1.y * WIDTH);
	const int coord2 = 3 * (p2.x + p2.y * WIDTH);
	
	//return maximum among three channels
	return max( abs_diff(img[coord1],     img[coord2]),
		   max( abs_diff(img[coord1 + 1], img[coord2 + 1]), 
		        abs_diff(img[coord1 + 2], img[coord2 + 2]) ) );
}

float DS(const int2 p1, const int2 p2) {
/* DS metric */
	return sqrt( (float) ( (p1.x - p2.x) * (p1.x - p2.x) + 
						   (p1.y - p2.y) * (p1.y - p2.y) ) );
}

bool supportRegionRule(__global uchar* img, const int2 keyPoint, const int2 borderPoint) {
/* using rule from the article, detect borders */

	if (DC(img, keyPoint, borderPoint) < TAU1 && DS(keyPoint, borderPoint) < L1) return true;
	else return false;
}

int detectBorderPixel(__global uchar* img, const int2 keyPoint, const int direction) {
/* detect support region fir everey pixel 
   0 - left direction 
   1 - right 
   2 - up 
   3 - down */

	int border;

	switch (direction) {
		case 0: {
			int counter = 1;
			
			while ( keyPoint.x - counter >= 0 && 
				    supportRegionRule(img, keyPoint, (int2)(keyPoint.x - counter))) counter++;
			
			border = keyPoint.x - counter + 1;
			
			break;
		}

		case 1: {
			int counter = 1;

			while (keyPoint.x + counter < WIDTH &&
				   supportRegionRule(img, keyPoint, (int2)(keyPoint.x + counter))) counter++;

			border = keyPoint.x + counter - 1;

			break;
		}

		case 2: {
			int counter = 1;

			while (keyPoint.y - counter >= 0 &&
				   supportRegionRule(img, keyPoint, (int2)(keyPoint.y - counter))) counter++;

			border = keyPoint.y - counter + 1;

			break;
		}

		case 3: {
			int counter = 1;

			while (keyPoint.y + counter < HEIGHT &&
				   supportRegionRule(img, keyPoint, (int2)(keyPoint.y + counter))) counter++;

			border = keyPoint.y + counter - 1;

			break;
		}
		default: 
			break;
	}

	return border;
}

__kernel void kDetectSupportRegions(__global uchar* lImg,
								    __global ushort* supportRegion) {
/* detect border pixels for every pixel (left-right-up-down) */

	const int3 xyz = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	// last argument is a variant of direction (0 - L, 1 - R, 2 - U, 3 - D)
	supportRegion[xyz.x + xyz.y * WIDTH + xyz.z * SQUARE] = detectBorderPixel(lImg, (int2)(xyz.x, xyz.y), xyz.z);
}