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

#include "astra/CudaForwardProjectionAlgorithm3DBatch.h"

#ifdef ASTRA_CUDA

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/CompositeGeometryManager.h"

#include "astra/Logging.h"

namespace astra {

CCudaForwardProjectionAlgorithm3DBatch::CCudaForwardProjectionAlgorithm3DBatch()
{
	m_bIsInitialized = false;
	m_iGPUIndex = -1;
	m_iDetectorSuperSampling = 1;
	m_pProjector = nullptr;
}

CCudaForwardProjectionAlgorithm3DBatch::~CCudaForwardProjectionAlgorithm3DBatch()
{
}

void CCudaForwardProjectionAlgorithm3DBatch::initializeFromProjector()
{
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector)
			ASTRA_WARN("non-CUDA Projector3D passed to FP3D_CUDA_BATCH");
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

bool CCudaForwardProjectionAlgorithm3DBatch::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaForwardProjectionAlgorithm3DBatch", this, _cfg);

	// Optional projector
	m_pProjector = nullptr;
	int id = -1;
	if (CR.has("ProjectorId")) {
		CR.getID("ProjectorId", id);
		m_pProjector = CProjector3DManager::getSingleton().get(id);
		if (!m_pProjector)
			ASTRA_WARN("Optional parameter ProjectorId is not a valid id");
	}

	std::vector<int> projIds;
	std::vector<int> volIds;

	bool ok = true;
	ok &= CR.getRequiredIntArray("ProjectionDataIds", projIds);
	ok &= CR.getRequiredIntArray("VolumeDataIds", volIds);
	if (!ok)
		return false;

	ASTRA_CONFIG_CHECK(!projIds.empty(), "FP3D_CUDA_BATCH", "No ProjectionDataIds specified.");
	ASTRA_CONFIG_CHECK(projIds.size() == volIds.size(), "FP3D_CUDA_BATCH",
	                   "ProjectionDataIds and VolumeDataIds must have equal length.");

	m_pProjections.clear();
	m_pVolumes.clear();
	m_pProjections.reserve(projIds.size());
	m_pVolumes.reserve(volIds.size());

	for (size_t i = 0; i < projIds.size(); ++i) {
		CFloat32ProjectionData3D* pProj =
		    dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(projIds[i]));
		CFloat32VolumeData3D* pVol =
		    dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(volIds[i]));

		ASTRA_CONFIG_CHECK(pProj, "FP3D_CUDA_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(pVol, "FP3D_CUDA_BATCH", "Invalid Volume Data Object.");

		m_pProjections.push_back(pProj);
		m_pVolumes.push_back(pVol);
	}

	initializeFromProjector();

	// Deprecated options (kept for compatibility with FP3D_CUDA)
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, m_iDetectorSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);
	if (!ok)
		return false;

	m_bIsInitialized = check();
	return m_bIsInitialized;
}

bool CCudaForwardProjectionAlgorithm3DBatch::check()
{
	ASTRA_CONFIG_CHECK(!m_pProjections.empty(), "FP3D_CUDA_BATCH", "No Projection Data Objects specified.");
	ASTRA_CONFIG_CHECK(m_pProjections.size() == m_pVolumes.size(), "FP3D_CUDA_BATCH",
	                   "Number of Projection and Volume Data Objects must match.");

	for (size_t i = 0; i < m_pProjections.size(); ++i) {
		ASTRA_CONFIG_CHECK(m_pProjections[i], "FP3D_CUDA_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(m_pVolumes[i], "FP3D_CUDA_BATCH", "Invalid Volume Data Object.");
		ASTRA_CONFIG_CHECK(m_pProjections[i]->isInitialized(), "FP3D_CUDA_BATCH", "Projection Data Object Not Initialized.");
		ASTRA_CONFIG_CHECK(m_pVolumes[i]->isInitialized(), "FP3D_CUDA_BATCH", "Volume Data Object Not Initialized.");
	}

	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "FP3D_CUDA_BATCH", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FP3D_CUDA_BATCH", "GPUIndex must be a non-negative integer.");

	m_bIsInitialized = true;
	return true;
}

bool CCudaForwardProjectionAlgorithm3DBatch::run(int)
{
	ASTRA_ASSERT(m_bIsInitialized);

	CCompositeGeometryManager cgm;
	CCompositeGeometryManager::TJobList jobs;

	for (size_t i = 0; i < m_pProjections.size(); ++i) {
		jobs.push_back(cgm.createJobFP(m_pProjector, m_pVolumes[i], m_pProjections[i],
		                               CCompositeGeometryManager::MODE_SET));
	}

	return cgm.doJobs(jobs);
}

} // namespace astra

#endif

