/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#ifndef _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM3DBATCH
#define _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM3DBATCH

#include "Globals.h"

#include "Algorithm.h"

#include "Data3D.h"

#ifdef ASTRA_CUDA

#include <vector>

namespace astra {

class CProjector3D;

// Batch wrapper around FP3D_CUDA. This reduces Python overhead by letting the
// C++ layer loop over multiple (vol, proj) pairs in one algorithm.run call.
class _AstraExport CCudaForwardProjectionAlgorithm3DBatch : public CAlgorithm
{
public:
	static inline const char* const type = "FP3D_CUDA_BATCH";

	CCudaForwardProjectionAlgorithm3DBatch();
	virtual ~CCudaForwardProjectionAlgorithm3DBatch();

	virtual bool initialize(const Config& _cfg);
	virtual bool run(int _iNrIterations = 0);
	virtual std::string description() const;

	bool check();

protected:
	void initializeFromProjector();

	CProjector3D* m_pProjector;
	std::vector<CFloat32ProjectionData3D*> m_pProjections;
	std::vector<CFloat32VolumeData3D*> m_pVolumes;

	// Deprecated options kept for config compatibility with FP3D_CUDA.
	int m_iGPUIndex;
	int m_iDetectorSuperSampling;
};

inline std::string CCudaForwardProjectionAlgorithm3DBatch::description() const { return CCudaForwardProjectionAlgorithm3DBatch::type; };

} // namespace astra

#endif

#endif

