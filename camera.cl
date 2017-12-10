const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
			CLK_ADDRESS_CLAMP_TO_EDGE |
			CLK_FILTER_NEAREST;

typedef struct tag_PixelBGR {
        unsigned char B;
        unsigned char G;
        unsigned char R;
}PixelBGR;


kernel void BGR2Gray(global PixelBGR *bgr, global uchar *gray) {
        uint pos = get_global_id(0);
        gray[pos] = bgr[pos].B * 0.114f + bgr[pos].G * 0.587f + bgr[pos].R * 0.299f;
}

//kernel void BGR2GrayImage(read_only image2d_t bgr, write_only image2d_t gray) {
//	int2 pos = (int2)(get_global_id(0), get_global_id(1));
//	uint4 color = read_imageui(bgr, sampler, pos);
//	write_imageui(gray, pos, color);
//	//uint _gray = (color.x * 0.114f + color.y * 0.587f + color.z * 0.299f);
//	//write_imageui(gray, pos, (uint4)(_gray, 0, 0, 0));
//}

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



kernel void Input2SDR(global uchar *Input, global uchar16 *SDR) {
        int pos = get_global_id(0);
        SDR[pos] = ucharToSDR(Input[pos]);

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

