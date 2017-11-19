// ===================
// Compute-program.h
// ===================

#ifndef COMPUTE_PROGRAM_H
#define COMPUTE_PROGRAM_H

#include "compute-system.h"

class ComputeProgram
{
	public:
		ComputeProgram(ComputeSystem &cs, const std::string &fileName);
		//bool loadFromFile(ComputeSystem &cs, const std::string &fileName);

		cl::Program getProgram()
		{
			return _program;
		}
	
	private:
		cl::Program _program;
};

#endif
