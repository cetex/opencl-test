kernel void Overlap(
		global uchar   *columns, 
		global uchar   *prevColumns,
		       uint2   columnDim, 
		       uint2   columnInputDim, 
		       uint2   columnSdrMult,
		global uchar16 *sdr, 
		       uint2   sdrDim, 
		global uchar16 *dendrites, 
		       uint2   dendritesDim,
		global uchar   *boost,
		global uchar   *distalDendrites,
		       uint2   distalDendritesDim)
{
	// Get global_id(0|1) is the position of the column.
	private uint2 colCenter = {get_global_id(0), get_global_id(1)};
	private uint2 colSize = {get_global_size(0), get_global_size(1)};

	// copy to local memory since it's supposed to be faster
	// Divide y by 16 since we're using uchar16
	sdrDim.y = sdrDim.y / 16;
	dendritesDim.y = dendritesDim.y/16;

	// Column dendrite threshold
	const uchar dendThresh = 127;
	
	// Store the number of active inputs
	private uchar active = 0;

	private uint2 sdrCenter = {
		colCenter.x / columnSdrMult.x,
		colCenter.y / columnSdrMult.y};

	private float2 dendStart = {
		colCenter.x * columnInputDim.x * dendritesDim.y,
		colCenter.y * columnInputDim.y};

	private int dendSize = dendritesDim.x * dendritesDim.y;

	for (private int x = 0; x < columnInputDim.x; x++) {
		for (private int y = 0; y< columnInputDim.y; y++) {
			
			private int sdrPos = (sdrCenter.x + x) * sdrDim.y + (sdrCenter.y + y);
			private int dendPos = dendStart.x + dendritesDim.y * x + dendStart.y +y;

			private uchar16 _active16 = sdr[sdrPos] & dendrites[dendPos];
			for (private int i = 0; i < 16; i++) 
				if (_active16[i] > dendThresh) active++;
		}
	}
	columns[colCenter.x * colSize.y + colCenter.y] = active + boost[colCenter.x * colSize.y + colCenter.y];
}


// We're launched as many times as it takes to find 2% of winning columns
// stepsize is columns dimensions * 0.02
kernel void Inhibit(
		global uchar *columns, 
		       uint2 columnsDim,
		       uint  stepsize) 
{
	// Get id
	private float id = get_global_id(0);

	// Store the winning active input
	private float winner = -1;

	// We're not taking spatial closeness into account, we're just finding the single winning column
	// through searching our small part of the column-space.
	private float startpos = stepsize * id;
	private float endpos   = stepsize * (id +1);
	// Make sure we're within bounds
	if (endpos >= columnsDim.x * columnsDim.y) {
		endpos = columnsDim.x * columnsDim.y;
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

kernel void Learn(
		global uchar   *columns, 
		global uchar   *prevColumns, 
		       uint2   columnDim, 
		       uint2   columnInputDim, 
		       uint2   columnSdrMult,
		global uchar16 *sdr, 
		       uint2   sdrDim, 
		global uchar16 *dendrites, 
		       uint2   dendritesDim,
		global uchar   *boost,
		global uchar   *distalDendrites,
		       uint2   distalDendritesDim)
{
	// Get global_id(0|1) is the position of the column.
	private uint2 colCenter = {get_global_id(0), get_global_id(1)};
	private uint2 colSize = {get_global_size(0), get_global_size(1)};
	
	// Exit early if we're not active.
	// Should implement boosting here.
	if (columns[colCenter.x * colSize.y + colCenter.y] == 0) {
	       boost[colCenter.x * colSize.y + colCenter.y] +=1;
       	       return;
	}
	boost[colCenter.x * colSize.y + colCenter.y] = 0;

	// copy to local memory since it's supposed to be faster
	// Divide y by 16 since we're using uchar16
	sdrDim.y = sdrDim.y / 16;
	dendritesDim.y = dendritesDim.y/16;

	private uint2 sdrCenter = {
		colCenter.x / columnSdrMult.x,
		colCenter.y / columnSdrMult.y};

	private float2 dendStart = {
		colCenter.x * columnInputDim.x * dendritesDim.y,
		colCenter.y * columnInputDim.y};

	private int dendSize = dendritesDim.x * dendritesDim.y;

	for (private int x = 0; x < columnInputDim.x; x++) {
		for (private int y = 0; y< columnInputDim.y; y++) {
			
			private int sdrPos = (sdrCenter.x + x) * sdrDim.y + (sdrCenter.y + y);
			private int dendPos = dendStart.x + dendritesDim.y * x + dendStart.y +y;

			private uchar16 _sdr16 = sdr[sdrPos];
			private uchar16 _dend16 = dendrites[dendPos];
			/*
			private uchar16 update = _dend16 > (uchar16){248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248} ? // are we close to overflowing
				_sdr16 & _dend16 : // close to overflowing, won't add more weight.
				_sdr16 & (_dend16 + (uchar16){10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}); // not overflowing, add weight.
			update = _dend16 < (uchar16){5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5} ? // are we close to underflowing?
				update : // close to underflowing, won't subtract more weights
				update + (~_sdr16 & (_dend16 - (uchar16){5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5})); // not underflowing, subtract more weight.
			*/
			for (private int i = 0; i < 16; i++) {
				//update[i] = _dend16[i] > 248 ? _sdr16[i] & _dend16[i] : _sdr16[i] & (_dend16[i] + 10);
				//update[i] = _dend16[i] < 5 ? update[i] : update[i] + (~_sdr16[i] & (_dend16[i] -5));
				if (_sdr16[i] > 0 && _dend16[i] < 248) {
					_dend16[i] += 5;
				}
				if (_sdr16[i] == 0 && _dend16[i] >5) {
					_dend16[i] -= 1;
				}
			}

			dendrites[dendPos] = _dend16;


		}
	}
	columns[colCenter.x * colSize.y + colCenter.y] = 255;
}
