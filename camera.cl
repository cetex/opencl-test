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

kernel void bufferbuffer(global uchar* image1, global uchar* image2) {
	uint pos = get_global_id(0);
	image2[pos] = image1[pos] - 128;
}

kernel void BGR2GrayImage(read_only image2d_t bgr, write_only image2d_t gray) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	uint4 color = read_imageui(bgr, sampler, pos);
	write_imageui(gray, pos, color);
	//uint _gray = (color.x * 0.114f + color.y * 0.587f + color.z * 0.299f);
	//write_imageui(gray, pos, (uint4)(_gray, 0, 0, 0));
}
