// ===============
// utils.h
// ===============


#ifndef UTILS_H
#define UTILS_H

class Vec2i
{
	public:
		int x, y;

		Vec2i()
			: x(16), y(16)
		{}

		Vec2i(int X, int Y)
			: x(X), y(Y)
		{}
		friend std::ostream& operator<<(std::ostream& os, const Vec2i& v) {
			os << v.x << "," << v.y;
			return os;
		}
};
#endif
