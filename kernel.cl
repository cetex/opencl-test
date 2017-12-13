ushort8 getNeuron(uint *pos, ushort8 *data) {
	return data[*pos];
	//(pixelHeight * neurons_dendrites_width) + (pixelWidth * 16)
}

uint getColumnRow(uint *row, uint *step, uint *col) {
	return (*row * *step) + (*col * sizeof(ushort8));
}

typedef struct nnDimensions {
	int columns_size_r;
	int columns_size_c;
	int columns_spacing;
	int sdr_size_r;
	int sdr_size_c;
	int sdr_per_column_r;
	int sdr_per_column_c;
	int dendrites_r;
	int dendrites_c;
	int inhibition_area_r;
	int inhibition_area_c;
	uchar synapse_threshold;
	uchar column_threshold;
	int numActiveColumnsPerInhArea;
} mynnDimensions;

uint getpixel(uint2 position, uint2 dimensions) {
	uint row = position.x * dimensions.y;
	uint col = position.y;
	return row + col;
}

kernel void CalculateOverlap(
		global uchar *columns, 
		global uint2 *columnDim, 
		global uint2 *columnInputDim, 
		global uchar *sdr, 
		global uint2 *sdrDim, 
		global uchar16 *dendrites, 
		global uint2 *dendritesDim) 
{
	// Get global_id(0|1) is the position of the column.
	private uint2 colCenter = {get_global_id(0), get_global_id(1)};
	private uint2 colSize = {get_global_size(0), get_global_size(1)};
	// Column threshold
	const uchar threshold = 127;

	// Store the number of active inputs
	private uchar active = 0;
	// Set column to off / black by default (DEBUG)
	columns[colCenter.x * colSize.y + colCenter.y] = 0;

	// Loop through all inputs (from SDR) and count the number of active.
	// dividing columnInputDim by 2 to center around middle.
	private int halfx = columnInputDim->x / 2;
	private int halfy = columnInputDim->y / 2;
	
	private int sdrMultX = sdrDim->x / columnDim->x;
	if (sdrMultX < 1) sdrMultX = 1;
	private int sdrMultY = sdrDim->y / columnDim->y;
	if (sdrMultY < 1) sdrMultY = 1;
	for (private int x = -halfx; x<=halfx-1; x++) {
		for (private int y = -halfy; y<=halfy-1; y++) {
			// Get y (columns)
			int pY = colCenter.y + y;
			// if y is negative, loop around from other end
			if (pY < 0) {
				pY = colSize.y + pY;
			}
			else if (pY >= colSize.y) {
				pY = pY - colSize.y;
			}

			// Get x (rows)
			int pX = colCenter.x + x;
			// if x is negative, loop around from other end
			if (pX < 0) {
				pX = colSize.x + pX;
			}
			else if (pX >= colSize.x) {
				pX = pX - colSize.x;
			}

			uint sdrPos = pX * sdrMultX * colSize.y + pY * sdrMultY;
			if (sdr[sdrPos] > 0) active++;

		}
	}
	if (active > threshold) {
		columns[colCenter.x * colSize.y + colCenter.y] = active;
	} else {
		columns[colCenter.x * colSize.y + colCenter.y] = 0;
	}
}

__kernel void CalculateOverlapOld(__global uchar16 *SDR, __global uchar16 *spatial_weights, __global uchar *columns, __global mynnDimensions *params) {
	__private uint2 col;
	col.x = get_global_id(0);
	col.y = get_global_id(1);
	
	__private uchar active_inputs;
	__private uchar16 Active;
	__private int c_spacing = params->columns_spacing;
	__private uchar synapse_threshold = params->synapse_threshold;

	__private int sdr_per_column_r = params->sdr_per_column_r;
	__private int sdr_per_column_c = params->sdr_per_column_c;
	__private int sdr_size_c = params->sdr_size_c;
	__private int dendrites_c = params->dendrites_c;


	for (__private int r = 0; r < sdr_per_column_r; r++) {
	__private int sdr_pos = ((col.x+r)*sdr_size_c)+col.y;
	__private int spatial_pos = ((col.x+r)*dendrites_c)+(col.y*sdr_per_column_c);
		for (__private int c = 0; c < sdr_per_column_c; c++) {
			//__private int sdr_pos = ((col.x+r)*sdr_size_c)+col.y+c;
			//__private int spatial_pos = ((col.x+r)*dendrites_c)+((col.y*sdr_per_column_c)+c);
			Active = SDR[sdr_pos+c] & spatial_weights[spatial_pos+c];
			for (__private int i = 0; i < 16; i++) {
				if (Active[i] > synapse_threshold) active_inputs++;
			}
		}
	}
	if (active_inputs < params->column_threshold) {
		//active_inputs = 0;
	}
	// Should probably add boost as another array of columns size here and multiply active_inputs with boost before saving.. 
	columns[(col.x*params->columns_size_c)+col.y] = active_inputs;

}
typedef struct weightpos {
	int x;
	uchar weight;
} weightpos;

__kernel void Inhibition(__global uchar *columns, __global uchar *columns_winners, __global mynnDimensions *params) {
	__private uint2 pos;
	pos.x = get_global_id(0);
	pos.y = get_global_id(1);
	//__local weightpos tmpColumns[params->columns_size_r*params->columns_size_c];
	
	//for (int i = 0; i < params->columns_size_r*params->columns_size_c; i++) {
	//	weightpos[i]->x = i;
	//	weightpos[i]->weight = columns_winners[i]; 
	//}
	// Here we need to loop through all 64 pixels in the SDR for this column and sum upp all active inputs.
	//__private uchar active_inputs;
	//__private uchar16 Active;
	//__private int c_spacing = params->columns_spacing; // private as it's reused a lot
	//for (int r = 0; r < params->inhibition_area_r; r++) {
	//	for (int c = 0; c < params->inhibition_area_c; c++) {
	//	
	//	}
	//}
}


