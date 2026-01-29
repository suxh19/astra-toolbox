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

#include "astra/CudaSirtAlgorithm3DBatch.h"

#ifdef ASTRA_CUDA

#include "astra/AstraObjectManager.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/CudaProjector3D.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"

namespace astra {

CCudaSirtAlgorithm3DBatch::CCudaSirtAlgorithm3DBatch()
{
	m_bIsInitialized = false;
	m_pProjector = nullptr;
	m_pSirt = nullptr;
	m_bAstraSIRTInit = false;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_fLambda = 1.0f;
}

CCudaSirtAlgorithm3DBatch::~CCudaSirtAlgorithm3DBatch()
{
	delete m_pSirt;
	m_pSirt = nullptr;
}

void CCudaSirtAlgorithm3DBatch::initializeFromProjector()
{
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector)
			ASTRA_WARN("non-CUDA Projector3D passed to SIRT3D_CUDA_BATCH");
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

bool CCudaSirtAlgorithm3DBatch::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaSirtAlgorithm3DBatch", this, _cfg);

	// Optional projector
	m_pProjector = nullptr;
	int projId = -1;
	if (CR.has("ProjectorId")) {
		CR.getID("ProjectorId", projId);
		m_pProjector = CProjector3DManager::getSingleton().get(projId);
		if (!m_pProjector)
			ASTRA_WARN("Optional parameter ProjectorId is not a valid id");
	}

	std::vector<int> sinoIds;
	std::vector<int> reconIds;

	bool ok = true;
	ok &= CR.getRequiredIntArray("ProjectionDataIds", sinoIds);
	ok &= CR.getRequiredIntArray("ReconstructionDataIds", reconIds);
	if (!ok)
		return false;

	ASTRA_CONFIG_CHECK(!sinoIds.empty(), "SIRT3D_CUDA_BATCH", "No ProjectionDataIds specified.");
	ASTRA_CONFIG_CHECK(sinoIds.size() == reconIds.size(), "SIRT3D_CUDA_BATCH",
	                   "ProjectionDataIds and ReconstructionDataIds must have equal length.");

	m_pSinograms.clear();
	m_pReconstructions.clear();
	m_pSinograms.reserve(sinoIds.size());
	m_pReconstructions.reserve(reconIds.size());

	for (size_t i = 0; i < sinoIds.size(); ++i) {
		CFloat32ProjectionData3D* pSino =
		    dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(sinoIds[i]));
		CFloat32VolumeData3D* pRecon =
		    dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(reconIds[i]));

		ASTRA_CONFIG_CHECK(pSino, "SIRT3D_CUDA_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(pRecon, "SIRT3D_CUDA_BATCH", "Invalid Reconstruction Data Object.");

		m_pSinograms.push_back(pSino);
		m_pReconstructions.push_back(pRecon);
	}

	ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);

	initializeFromProjector();

	// Deprecated options
	ok &= CR.getOptionInt("VoxelSuperSampling", m_iVoxelSuperSampling, m_iVoxelSuperSampling);
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, m_iDetectorSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, m_iGPUIndex);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, m_iGPUIndex);
	if (!ok)
		return false;

	// Lazily initialize the GPU-side SIRT instance at first run()
	delete m_pSirt;
	m_pSirt = new AstraSIRT3d();
	m_bAstraSIRTInit = false;

	m_bIsInitialized = check();
	return m_bIsInitialized;
}

bool CCudaSirtAlgorithm3DBatch::check()
{
	ASTRA_CONFIG_CHECK(!m_pSinograms.empty(), "SIRT3D_CUDA_BATCH", "No Projection Data Objects specified.");
	ASTRA_CONFIG_CHECK(m_pSinograms.size() == m_pReconstructions.size(), "SIRT3D_CUDA_BATCH",
	                   "Number of Projection and Reconstruction Data Objects must match.");

	ASTRA_CONFIG_CHECK(m_iVoxelSuperSampling >= 1, "SIRT3D_CUDA_BATCH", "VoxelSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "SIRT3D_CUDA_BATCH", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "SIRT3D_CUDA_BATCH", "GPUIndex must be a non-negative integer.");

	// Geometry and memory checks. We reuse one GPU buffer set across the batch,
	// so all items must share the same geometries.
	const CProjectionGeometry3D& projgeom0 = m_pSinograms[0]->getGeometry();
	const CVolumeGeometry3D& volgeom0 = m_pReconstructions[0]->getGeometry();

	for (size_t i = 0; i < m_pSinograms.size(); ++i) {
		CFloat32ProjectionData3D* sino = m_pSinograms[i];
		CFloat32VolumeData3D* recon = m_pReconstructions[i];

		ASTRA_CONFIG_CHECK(sino, "SIRT3D_CUDA_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(recon, "SIRT3D_CUDA_BATCH", "Invalid Reconstruction Data Object.");

		ASTRA_CONFIG_CHECK(sino->isInitialized(), "SIRT3D_CUDA_BATCH", "Projection Data Object Not Initialized.");
		ASTRA_CONFIG_CHECK(recon->isInitialized(), "SIRT3D_CUDA_BATCH", "Reconstruction Data Object Not Initialized.");

		ASTRA_CONFIG_CHECK(sino->isFloat32Memory(), "SIRT3D_CUDA_BATCH",
		                   "Projection Data Object must be float32/memory (CPU).");
		ASTRA_CONFIG_CHECK(recon->isFloat32Memory(), "SIRT3D_CUDA_BATCH",
		                   "Reconstruction Data Object must be float32/memory (CPU).");

		ASTRA_CONFIG_CHECK(sino->getGeometry().isEqual(&projgeom0), "SIRT3D_CUDA_BATCH",
		                   "All ProjectionDataIds must have identical geometry.");
		ASTRA_CONFIG_CHECK(recon->getGeometry().isEqual(&volgeom0), "SIRT3D_CUDA_BATCH",
		                   "All ReconstructionDataIds must have identical geometry.");
	}

	m_bIsInitialized = true;
	return true;
}

bool CCudaSirtAlgorithm3DBatch::run(int _iNrIterations)
{
	ASTRA_ASSERT(m_bIsInitialized);
	ASTRA_CONFIG_CHECK(m_pSirt, "SIRT3D_CUDA_BATCH", "Internal error: SIRT instance not initialized.");

	const CProjectionGeometry3D& projgeom0 = m_pSinograms[0]->getGeometry();
	const CVolumeGeometry3D& volgeom0 = m_pReconstructions[0]->getGeometry();

	bool ok = true;

	if (!m_bAstraSIRTInit) {
		ok &= m_pSirt->setGPUIndex(m_iGPUIndex);
		ok &= m_pSirt->setGeometry(&volgeom0, &projgeom0);
		ok &= m_pSirt->enableSuperSampling(m_iVoxelSuperSampling, m_iDetectorSuperSampling);
		ok &= m_pSirt->setRelaxation(m_fLambda);
		ok &= m_pSirt->init();

		ASTRA_ASSERT(ok);
		m_bAstraSIRTInit = true;
	}

	for (size_t i = 0; i < m_pSinograms.size(); ++i) {
		CFloat32ProjectionData3D* sino = m_pSinograms[i];
		CFloat32VolumeData3D* recon = m_pReconstructions[i];

		ASTRA_ASSERT(sino->isFloat32Memory());
		ASTRA_ASSERT(recon->isFloat32Memory());

		ok = m_pSirt->setSinogram(sino->getFloat32Memory(), sino->getGeometry().getDetectorColCount());
		ASTRA_ASSERT(ok);

		ok = m_pSirt->setStartReconstruction(recon->getFloat32Memory(), volgeom0.getGridColCount());
		ASTRA_ASSERT(ok);

		ok = m_pSirt->iterate(_iNrIterations);
		ASTRA_ASSERT(ok);

		ok = m_pSirt->getReconstruction(recon->getFloat32Memory(), volgeom0.getGridColCount());
		ASTRA_ASSERT(ok);
	}

	return ok;
}

} // namespace astra

#endif

