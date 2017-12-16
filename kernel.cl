kernel void Overlap(
		global uchar   *columns, 
		global uint2   *columnDim, 
		global uint2   *columnInputDim, 
		global uchar   *sdr, 
		global uint2   *sdrDim, 
		global uchar   *dendrites, 
		global uint2   *dendritesDim) 
{
	// Get global_id(0|1) is the position of the column.
	private uint2 colCenter = {get_global_id(0), get_global_id(1)};
	private uint2 colSize = {get_global_size(0), get_global_size(1)};

	// copy to local memory since it's supposed to be faster
	private uint2 _sdrDim = {
		sdrDim->x,
		sdrDim->y};

	private uint2 _dendritesDim = {
		dendritesDim->x,
		dendritesDim->y};

	// Column dendrite threshold
	const uchar dendThresh = 127;
	
	// Store the number of active inputs
	private uchar active = 0;

	private int2 sdrMult = {
		_sdrDim.x / columnDim->x,
		_sdrDim.y / columnDim->y};

	// DEBUG, set column pixels to black
	columns[colCenter.x * colSize.y + colCenter.y] = 0;

	private int2 sdrStart = {
		colCenter.x * sdrMult.x,
		colCenter.y * sdrMult.y};
	
	private int2 dendStart = {
		colCenter.x * columnInputDim->x * _dendritesDim.y,
		colCenter.y * columnInputDim->y};

	for (private int x = 0; x <= columnInputDim->x; x++) {
		for (private int y = 0; y<= columnInputDim->y; y++) {
			// SDR may be smaller than columns, therefore we need to
			// check if we're out of bounds (on the larger side)
			private int2 sdrXY = {
				sdrStart.x + x,
				sdrStart.y + y};
			if (sdrXY.x >= _sdrDim.x) {
				// out of bounds (expected for columns close to bottom)
				continue;

			}
			if (sdrXY.y >= _sdrDim.y) {
				// out of bounds (expected for columns close to right boundary)
				continue;
			}

			uint sdrPos = sdrXY.x * _sdrDim.y + sdrXY.y;
			uint dendPos = dendStart.x + _dendritesDim.y * x + dendStart.y + y;
			if (sdr[sdrPos] > 0 && dendrites[dendPos] > dendThresh) active++;
		}
	}
	columns[colCenter.x * colSize.y + colCenter.y] = active;
}


// We're launched as many times as it takes to find 2% of winning columns
// stepsize is columns dimensions * 0.02
kernel void Inhibit(
		global uchar *columns, 
		global uint2 *columnsDim,
		global uint  *stepsize) 
{
	// Get id
	private uint id = get_global_id(0);

	// Store the winning active input
	private int winner = -1;

	// We're not taking spatial closeness into account, we're just finding the single winning column
	// through searching our small part of the column-space.
	private uint startpos = (*stepsize) * id;
	private uint endpos   = (*stepsize) * (id +1);
	// Make sure we're within bounds
	if (endpos >= columnsDim->x * columnsDim->y) {
		endpos = columnsDim->x * columnsDim->y;
	}

	// Loop through all columns from startpos to endpos, find winner, set everyone else to 0
	for (private int i = startpos; i <= endpos; i++) {
		if (columns[i] > winner)
		{
			winner = columns[i];
		};
	}

	for (private int i = startpos; i <= endpos; i++) {
		if (columns[i] < winner)
	       	{
			columns[i] = 0;
		}
	}

}
//kernel void Inhibit(
//		global uchar *columns, 
//		global uint2 *columnsDim,
//		global uint  *stepsize) 
//{
//	// Get id
//	private uint id = get_global_id(0);
//
//	// Store the winning active input
//	private int winner = -1;
//
//	// We're not taking spatial closeness into account, we're just finding the single winning column
//	// through searching our small part of the column-space.
//	private uint startpos = *stepsize * id;
//	private uint endpos   = *stepsize * (id +1);
//	// Loop through all columns from startpos to endpos, find winner, set everyone else to 0
//	for (private int i = startpos; i < endpos; i++) {
//		if (columns[i] > winner)
//		{
//			winner = columns[i];
//		};
//	}
//
//	for (private int i = startpos; i < endpos; i++) {
//		if (columns[i] < winner)
//	       	{
//			columns[i] = 0;
//		}
//	}
//
//}

kernel void Learn(
		global uchar   *columns, 
		global uint2   *columnDim, 
		global uint2   *columnInputDim, 
		global uchar   *sdr, 
		global uint2   *sdrDim, 
		global uchar   *dendrites, 
		global uint2   *dendritesDim) 
{
	// Get global_id(0|1) is the position of the column.
	private uint2 colCenter = {get_global_id(0), get_global_id(1)};
	private uint2 colSize = {get_global_size(0), get_global_size(1)};
	
	// Exit early if we're not active.
	// Should implement boosting here.
	if (columns[colCenter.x * colSize.y + colCenter.y] == 0) return;

	private int sdrMultX = sdrDim->x / columnDim->x;
	private int sdrMultY = sdrDim->y / columnDim->y;

	// DEBUG, set column pixels to black
	columns[colCenter.x * colSize.y + colCenter.y] = 0;

	private int2 dendStart = {
		colCenter.x * columnInputDim->x * dendritesDim->y,
		colCenter.y * columnInputDim->y};

	for (private int x = 0; x <= columnInputDim->x; x++) {
		for (private int y = 0; y<= columnInputDim->y; y++) {
			// SDR may be smaller than columns, therefore we need to
			// check if we're out of bounds (on the larger side)
			uint sdrX = colCenter.x * sdrMultX + x;
			if (sdrX > sdrDim->x) {
				sdrX = sdrX - sdrDim->x;
			}
			uint sdrY = colCenter.y * sdrMultY + y;
			if (sdrY > sdrDim->y) {
				sdrY = sdrY - sdrDim->y;
			}

			//uint sdrPos = sdrStart.x + sdrDim->y * x + sdrStart.y + y;
			uint sdrPos = sdrX * sdrDim->y + sdrY;
			uint dendPos = dendStart.x + dendritesDim->y * x + dendStart.y + y;
			if (sdr[sdrPos] > 0) {
				// Synapse is active, increase permanence by 10%
				dendrites[dendPos] += 10;
			}
		}
	}
	columns[colCenter.x * colSize.y + colCenter.y] = 255;
}

kernel void Forget(global uchar *dendrites) 
{
	// Get id
	private uint2 id = {get_global_id(0), get_global_id(1)};
	dendrites[get_global_id(0) * get_global_size(1) + get_global_id(1)] -= 5;
}

