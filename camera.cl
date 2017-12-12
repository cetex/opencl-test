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
