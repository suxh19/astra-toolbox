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

#include "astra/CudaSirtAlgorithm3DGpuBatch.h"

#ifdef ASTRA_CUDA

#include <cuda_runtime_api.h>

#include <atomic>
#include <thread>

#include "astra/AstraObjectManager.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/CudaProjector3D.h"

#include "astra/GeometryUtil3D.h"
#include "astra/Logging.h"

#include "astra/cuda/3d/mem3d.h"
#include "astra/cuda/3d/sirt3d.h"

namespace astra {

CCudaSirtAlgorithm3DGpuBatch::CCudaSirtAlgorithm3DGpuBatch()
{
	m_bIsInitialized = false;
	m_pProjector = nullptr;
	m_bSirtInit = false;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_fLambda = 1.0f;
	m_iNumWorkers = 1;
}

CCudaSirtAlgorithm3DGpuBatch::~CCudaSirtAlgorithm3DGpuBatch()
{
}

void CCudaSirtAlgorithm3DGpuBatch::initializeFromProjector()
{
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector)
			ASTRA_WARN("non-CUDA Projector3D passed to SIRT3D_CUDA_GPU_BATCH");
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

bool CCudaSirtAlgorithm3DGpuBatch::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaSirtAlgorithm3DGpuBatch", this, _cfg);

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

	ASTRA_CONFIG_CHECK(!sinoIds.empty(), "SIRT3D_CUDA_GPU_BATCH", "No ProjectionDataIds specified.");
	ASTRA_CONFIG_CHECK(sinoIds.size() == reconIds.size(), "SIRT3D_CUDA_GPU_BATCH",
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

		ASTRA_CONFIG_CHECK(pSino, "SIRT3D_CUDA_GPU_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(pRecon, "SIRT3D_CUDA_GPU_BATCH", "Invalid Reconstruction Data Object.");

		m_pSinograms.push_back(pSino);
		m_pReconstructions.push_back(pRecon);
	}

	ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);
	ok &= CR.getOptionInt("NumWorkers", m_iNumWorkers, 1);

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

	if (m_iNumWorkers < 1)
		m_iNumWorkers = 1;

	// Lazily initialize the GPU-side SIRT instances at first run()
	m_pSirtWorkers.clear();
	m_pSirtWorkers.reserve(m_iNumWorkers);
	for (int i = 0; i < m_iNumWorkers; ++i)
		m_pSirtWorkers.emplace_back(std::make_unique<astraCUDA3d::SIRT>());
	m_bSirtInit = false;

	m_bIsInitialized = check();
	return m_bIsInitialized;
}

bool CCudaSirtAlgorithm3DGpuBatch::check()
{
	ASTRA_CONFIG_CHECK(!m_pSinograms.empty(), "SIRT3D_CUDA_GPU_BATCH", "No Projection Data Objects specified.");
	ASTRA_CONFIG_CHECK(m_pSinograms.size() == m_pReconstructions.size(), "SIRT3D_CUDA_GPU_BATCH",
	                   "Number of Projection and Reconstruction Data Objects must match.");
	ASTRA_CONFIG_CHECK(m_iNumWorkers >= 1, "SIRT3D_CUDA_GPU_BATCH", "NumWorkers must be a positive integer.");

	ASTRA_CONFIG_CHECK(m_iVoxelSuperSampling >= 1, "SIRT3D_CUDA_GPU_BATCH", "VoxelSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "SIRT3D_CUDA_GPU_BATCH", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "SIRT3D_CUDA_GPU_BATCH", "GPUIndex must be a non-negative integer.");

	const CProjectionGeometry3D& projgeom0 = m_pSinograms[0]->getGeometry();
	const CVolumeGeometry3D& volgeom0 = m_pReconstructions[0]->getGeometry();

	for (size_t i = 0; i < m_pSinograms.size(); ++i) {
		CFloat32ProjectionData3D* sino = m_pSinograms[i];
		CFloat32VolumeData3D* recon = m_pReconstructions[i];

		ASTRA_CONFIG_CHECK(sino, "SIRT3D_CUDA_GPU_BATCH", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(recon, "SIRT3D_CUDA_GPU_BATCH", "Invalid Reconstruction Data Object.");

		ASTRA_CONFIG_CHECK(sino->isInitialized(), "SIRT3D_CUDA_GPU_BATCH", "Projection Data Object Not Initialized.");
		ASTRA_CONFIG_CHECK(recon->isInitialized(), "SIRT3D_CUDA_GPU_BATCH", "Reconstruction Data Object Not Initialized.");

		ASTRA_CONFIG_CHECK(sino->getStorage()->isGPU(), "SIRT3D_CUDA_GPU_BATCH",
		                   "Projection Data Object must be float32/GPU.");
		ASTRA_CONFIG_CHECK(recon->getStorage()->isGPU(), "SIRT3D_CUDA_GPU_BATCH",
		                   "Reconstruction Data Object must be float32/GPU.");

		// SIRT3D requires linear pitched memory, not CUDA arrays.
		CDataGPU* sinoStorage = dynamic_cast<CDataGPU*>(sino->getStorage());
		CDataGPU* reconStorage = dynamic_cast<CDataGPU*>(recon->getStorage());
		ASTRA_CONFIG_CHECK(sinoStorage, "SIRT3D_CUDA_GPU_BATCH", "Invalid GPU storage for sinogram.");
		ASTRA_CONFIG_CHECK(reconStorage, "SIRT3D_CUDA_GPU_BATCH", "Invalid GPU storage for reconstruction.");

		astraCUDA3d::MemPitchedPtrInfo3D info;
		ASTRA_CONFIG_CHECK(astraCUDA3d::getPitchedPtrInfo(sinoStorage->getHandle(), info),
		                   "SIRT3D_CUDA_GPU_BATCH", "Sinogram GPU storage must be a pitched pointer (not a CUDA array).");
		ASTRA_CONFIG_CHECK(astraCUDA3d::getPitchedPtrInfo(reconStorage->getHandle(), info),
		                   "SIRT3D_CUDA_GPU_BATCH", "Reconstruction GPU storage must be a pitched pointer (not a CUDA array).");

		ASTRA_CONFIG_CHECK(sino->getGeometry().isEqual(&projgeom0), "SIRT3D_CUDA_GPU_BATCH",
		                   "All ProjectionDataIds must have identical geometry.");
		ASTRA_CONFIG_CHECK(recon->getGeometry().isEqual(&volgeom0), "SIRT3D_CUDA_GPU_BATCH",
		                   "All ReconstructionDataIds must have identical geometry.");
	}

	m_bIsInitialized = true;
	return true;
}

bool CCudaSirtAlgorithm3DGpuBatch::run(int _iNrIterations)
{
	ASTRA_ASSERT(m_bIsInitialized);
	ASTRA_CONFIG_CHECK(!m_pSirtWorkers.empty(), "SIRT3D_CUDA_GPU_BATCH",
	                   "Internal error: SIRT instances not initialized.");

	const CProjectionGeometry3D& projgeom0 = m_pSinograms[0]->getGeometry();
	const CVolumeGeometry3D& volgeom0 = m_pReconstructions[0]->getGeometry();

	bool ok = true;

	if (!m_bSirtInit) {
		ok &= astraCUDA3d::setGPUIndex(m_iGPUIndex);
		ASTRA_CONFIG_CHECK(ok, "SIRT3D_CUDA_GPU_BATCH", "Failed to set GPU index.");

		astra::SDimensions3D dims;
		ok &= astra::convertAstraGeometry_dims(&volgeom0, &projgeom0, dims);
		ASTRA_CONFIG_CHECK(ok, "SIRT3D_CUDA_GPU_BATCH", "Failed to convert geometry.");

		astraCUDA3d::SProjectorParams3D params;
		params.iRaysPerVoxelDim = m_iVoxelSuperSampling;
		params.iRaysPerDetDim = m_iDetectorSuperSampling;
		params.fOutputScale = 1.0f;
		params.projKernel = astraCUDA3d::ker3d_default;

		astra::Geometry3DParameters projs = astra::convertAstraGeometry(&volgeom0, &projgeom0, params.volScale);
		ASTRA_CONFIG_CHECK(projs.isValid(), "SIRT3D_CUDA_GPU_BATCH", "Failed to convert geometry.");

		for (auto& sirt : m_pSirtWorkers) {
			ok &= sirt->setGeometry(dims, projs, params);
			ASTRA_CONFIG_CHECK(ok, "SIRT3D_CUDA_GPU_BATCH", "Failed to set SIRT geometry.");
			sirt->setRelaxation(m_fLambda);
			ok &= sirt->init();
			ASTRA_CONFIG_CHECK(ok, "SIRT3D_CUDA_GPU_BATCH", "Failed to initialize SIRT.");
		}

		m_bSirtInit = true;
	}

	const size_t n = m_pSinograms.size();
	const size_t workers = std::min(n, static_cast<size_t>(std::max(1, m_iNumWorkers)));

	std::atomic<bool> okAtomic{true};

	auto run_one = [&](astraCUDA3d::SIRT& sirt, size_t i) {
		CFloat32ProjectionData3D* sino = m_pSinograms[i];
		CFloat32VolumeData3D* recon = m_pReconstructions[i];

		CDataGPU* sinoStorage = dynamic_cast<CDataGPU*>(sino->getStorage());
		CDataGPU* reconStorage = dynamic_cast<CDataGPU*>(recon->getStorage());
		ASTRA_ASSERT(sinoStorage);
		ASTRA_ASSERT(reconStorage);

		astraCUDA3d::MemPitchedPtrInfo3D sinoInfo;
		astraCUDA3d::MemPitchedPtrInfo3D volInfo;
		bool lok = true;
		lok &= astraCUDA3d::getPitchedPtrInfo(sinoStorage->getHandle(), sinoInfo);
		lok &= astraCUDA3d::getPitchedPtrInfo(reconStorage->getHandle(), volInfo);
		ASTRA_ASSERT(lok);

		cudaPitchedPtr D_sino;
		D_sino.ptr = sinoInfo.ptr;
		D_sino.pitch = sinoInfo.pitch;
		D_sino.xsize = sinoInfo.xsize;
		D_sino.ysize = sinoInfo.ysize;

		cudaPitchedPtr D_vol;
		D_vol.ptr = volInfo.ptr;
		D_vol.pitch = volInfo.pitch;
		D_vol.xsize = volInfo.xsize;
		D_vol.ysize = volInfo.ysize;

		lok &= sirt.setBuffers(D_vol, D_sino);
		ASTRA_ASSERT(lok);
		lok &= sirt.iterate(_iNrIterations);
		ASTRA_ASSERT(lok);

		if (!lok)
			okAtomic.store(false, std::memory_order_relaxed);
	};

	if (workers <= 1) {
		for (size_t i = 0; i < n; ++i) {
			run_one(*m_pSirtWorkers[0], i);
			if (!okAtomic.load(std::memory_order_relaxed))
				break;
		}
	} else {
		std::vector<std::thread> threads;
		threads.reserve(workers);

		for (size_t t = 0; t < workers; ++t) {
			threads.emplace_back([&, t]() {
				// Each host thread needs to select the device before using CUDA.
				if (!astraCUDA3d::setGPUIndex(m_iGPUIndex)) {
					okAtomic.store(false, std::memory_order_relaxed);
					return;
				}

				astraCUDA3d::SIRT& sirt = *m_pSirtWorkers[t];
				for (size_t i = t; i < n; i += workers) {
					if (!okAtomic.load(std::memory_order_relaxed))
						break;
					run_one(sirt, i);
				}
			});
		}

		for (auto& th : threads)
			th.join();
	}

	ok &= okAtomic.load(std::memory_order_relaxed);
	return ok;
}

} // namespace astra

#endif
