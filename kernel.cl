#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct tag_PixelBGR {
	unsigned char B;
	unsigned char G;
	unsigned char R;
}PixelBGR;


__kernel void BGR2GRAY(__global PixelBGR *bgr, __global uchar *gray) {
	int pos = get_global_id(0);
	gray[pos] = bgr[pos].B * 0.114f + bgr[pos].G * 0.587f + bgr[pos].R * 0.299f;
}



uchar16 ucharToSDR(uchar data) {
	if (data < 128) {
		if (data < 63) {
			if (data < 32) {
				if (data < 16) {
					//0-15    = 0b0000000000000011
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0);
				} else {
					//16-31   = 0b0000000000000110
					//return 0b0000000000000110;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0);
				}
			} else {
				if (data < 48) {
					//32-47   = 0b0000000000001100
					//return 0b0000000000001100;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0);
					
				} else {
					//48-63   = 0b0000000000011000
					//return 0b0000000000011000;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0);
				}
			}
		} else {
			if (data < 96) {
				if (data < 80) {
					//64-79   = 0b0000000000110000
					//return 0b0000000000110000;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0);
				} else {
					//80-95   = 0b0000000001100000
					//return 0b0000000001100000;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0);
				}
			} else {
				if (data < 112) {
					//96-111  = 0b0000000011000000
					//return 0b0000000011000000;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0);
				} else {
					//112-127 = 0b0000000110000000
					//return 0b0000000110000000;
					return (uchar16)(0, 0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0);
				}
			}
		}
	} else {
		if (data < 192) {
			if (data < 160) {
				if (data < 144) {
					//128-143 = 0b0000001100000000
					//return 0b0000001100000000;
					return (uchar16)(0, 0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0);
				} else {
					//144-159 = 0b0000011000000000
					//return 0b0000011000000000;
					return (uchar16)(0, 0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				}
			} else {
				if (data < 176) {
					//160-175 = 0b0000110000000000
					//return 0b0000110000000000;
					return (uchar16)(0, 0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				} else {
					//176-191 = 0b0001100000000000
					//return 0b0001100000000000;
					return (uchar16)(0, 0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				}
			}
		} else {
			if (data < 224) {
				if (data < 208) {
					//192-207 = 0b0011000000000000
					//return 0b0011000000000000;
					return (uchar16)(0, 0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				} else {
					//208-223 = 0b0110000000000000
					//return 0b0110000000000000;
					return (uchar16)(0, ~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				}
			} else {
				if (data < 240) {
					//224-239 = 0b1100000000000000
					//return 0b1100000000000000;
					return (uchar16)(~0, ~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				} else {
					//240-256 = 0b1000000000000001
					//return 0b100000000000001;
					return (uchar16)(~0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				}
			}
		}
	}
}


__kernel void BGR2SDR(__global uchar *gray, __global uchar16 *SDR) {
	int pos = get_global_id(0);
	//__local uchar4 lsdr;
	//lsdr = SDR[pos.x*pos.y/4];

	//__local uchar4 data;
	//data.x = ucharToSDR(gray[pos.x * pos.y]);
	//data.y = ucharToSDR(gray[pos.x * pos.y + 1]);
	//data.z = ucharToSDR(gray[pos.x * pos.y + 2]);
	//data.w = ucharToSDR(gray[pos.x * pos.y + 3]);
	
	SDR[pos] = ucharToSDR(gray[pos]);



	//uchar gray = bgr[pos.x * pos.y].B * 0.114f + bgr[pos.x * pos.y].G * 0.587f + bgr[pos.x * pos.y].R * 0.299f;
	//0-15    = 0b0000000000000011
	//16-31   = 0b0000000000000110
	//32-47   = 0b0000000000001100
	//48-63   = 0b0000000000011000
	//64-79   = 0b0000000000110000
	//80-95   = 0b0000000001100000
	//96-111  = 0b0000000011000000
	//112-127 = 0b0000000110000000
	//128-143 = 0b0000001100000000
	//144-159 = 0b0000011000000000
	//160-175 = 0b0000110000000000
	//176-191 = 0b0001100000000000
	//192-207 = 0b0011000000000000
	//208-223 = 0b0110000000000000
	//224-239 = 0b1100000000000000
	//240-256 = 0b1000000000000001
}

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

__kernel void ColumnOverlap(__global uchar16 *SDR, __global uchar16 *spatial_weights, __global uchar *columns, __global mynnDimensions *params) {
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


