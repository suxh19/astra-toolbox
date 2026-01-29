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

#ifndef _INC_ASTRA_CUDASIRTALGORITHM3DGPUBATCH
#define _INC_ASTRA_CUDASIRTALGORITHM3DGPUBATCH

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"
#include "Data3D.h"

#ifdef ASTRA_CUDA

#include <vector>

namespace astraCUDA3d {
class SIRT;
}

namespace astra {

class CProjector3D;

// GPU-data variant of SIRT3D_CUDA_BATCH. This algorithm expects the input
// sinograms and output reconstructions to be stored on GPU (e.g. linked from
// DLPack/GPULink) and runs without host-device copies.
class _AstraExport CCudaSirtAlgorithm3DGpuBatch : public CAlgorithm
{
public:
	static inline const char* const type = "SIRT3D_CUDA_GPU_BATCH";

	CCudaSirtAlgorithm3DGpuBatch();
	virtual ~CCudaSirtAlgorithm3DGpuBatch();

	virtual bool initialize(const Config& _cfg);
	virtual bool run(int _iNrIterations = 0);
	virtual std::string description() const;

	bool check();

protected:
	void initializeFromProjector();

	CProjector3D* m_pProjector;
	std::vector<CFloat32ProjectionData3D*> m_pSinograms;
	std::vector<CFloat32VolumeData3D*> m_pReconstructions;

	astraCUDA3d::SIRT* m_pSirt;
	bool m_bSirtInit;

	int m_iGPUIndex;
	int m_iDetectorSuperSampling;
	int m_iVoxelSuperSampling;
	float m_fLambda;
};

inline std::string CCudaSirtAlgorithm3DGpuBatch::description() const { return CCudaSirtAlgorithm3DGpuBatch::type; };

} // namespace astra

#endif

#endif
