// ===============
// utils.h
// ===============


#ifndef UTILS_H
#define UTILS_H
#include "../compute/compute-system.h"
#include "../compute/compute-program.h"

#include <iostream>

namespace HTM {

class Vec2i
{
	public:
		unsigned int x, y;

		Vec2i()
			: x(16), y(16)
		{}

		Vec2i(unsigned int X, unsigned int Y)
			: x(X), y(Y)
		{}
		friend std::ostream& operator<<(std::ostream& os, const Vec2i& v) {
			os << v.x << "," << v.y;
			return os;
		}
};

class DoubleBuffer
{
	public:
		cl::Buffer *buffer;
		cl::Buffer *prevBuffer;
		Vec2i size;
		void setBuffer(cl::Buffer *newBuffer) {
			prevBuffer = buffer;
			buffer = newBuffer;
		}
		void flipBuffer() {
			cl::Buffer *tmpBuf = buffer;
			buffer = prevBuffer;
			prevBuffer = tmpBuf;
		}
};

DoubleBuffer* createDoubleBuffer(ComputeSystem &cs, Vec2i size);
};
#endif
