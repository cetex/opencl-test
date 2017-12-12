// ===============
// utils.cpp
// ===============

#include "utils.h"

namespace HTM {

DoubleBuffer* createDoubleBuffer(ComputeSystem &cs, Vec2i size) {
	DoubleBuffer *buf = new DoubleBuffer;
	buf->buffer = new cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE,
			size.x * size.y * sizeof(uint8_t), NULL, NULL);
	buf->prevBuffer = new cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE,
			size.x * size.y * sizeof(uint8_t), NULL, NULL);
	buf->size.x = size.x;
	buf->size.y = size.y;
	return buf;
}

};
