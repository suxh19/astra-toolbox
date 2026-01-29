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

#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include <string>
#include <optional>
#include <vector>
#include <tuple>

#include "astra/Globals.h"

#include "dims.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace astraCUDA {

bool copyVolumeToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch);
bool copyVolumeFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch);
bool copySinogramFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch);
bool copySinogramToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch);

bool allocateVolume(float*& D_ptr, unsigned int width, unsigned int height, unsigned int& pitch);
bool zeroVolume(float* D_data, unsigned int pitch, unsigned int width, unsigned int height, std::optional<cudaStream_t> _stream = {});

bool allocateVolumeData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims);
bool allocateProjectionData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims);
bool zeroVolumeData(float* D_ptr, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
bool zeroProjectionData(float* D_ptr, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});

bool duplicateVolumeData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
bool duplicateProjectionData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});

bool createArrayAndTextureObject2D(float* data, cudaArray*& dataArray, cudaTextureObject_t& texObj, unsigned int pitch, unsigned int width, unsigned int height, std::optional<cudaStream_t> _stream = {});
bool createTextureObjectPitch2D(float* D_data, cudaTextureObject_t& texObj, unsigned int pitch, unsigned int width, unsigned int height, cudaTextureAddressMode mode = cudaAddressModeBorder);


bool checkCuda(cudaError_t err, const char *msg);

float dotProduct2D(float* D_data, unsigned int pitch,
                   unsigned int width, unsigned int height,
                   std::optional<cudaStream_t> _stream = {});


// When no stream is provided, many helper functions used to create/destroy a
// temporary CUDA stream for every call. That overhead becomes significant in
// iterative 3D algorithms (e.g. SIRT/CGLS) that call these helpers frequently.
// We therefore keep a per-thread cached stream and synchronize it at the end
// of each helper call to preserve the original synchronous semantics.
struct CachedStream {
	int device = -1;
	cudaStream_t stream = 0;

	bool ensure() {
		int dev = 0;
		if (!checkCuda(cudaGetDevice(&dev), "StreamHelper getDevice"))
			return false;

		if (device != dev || !stream) {
			if (stream) {
				cudaStreamDestroy(stream);
				stream = 0;
			}
			device = dev;
			return checkCuda(cudaStreamCreate(&stream), "StreamHelper create");
		}

		return true;
	}

	~CachedStream() {
		if (stream)
			cudaStreamDestroy(stream);
	}
};

inline thread_local CachedStream g_cachedStream;


// Helper class for functions taking a std::optional<cudaStream_t> argument.
// If a stream isn't passed to us, use a per-thread cached stream and
// synchronize it at the end of the call.
class StreamHelper {
public:
	StreamHelper(std::optional<cudaStream_t> _stream) {
		ok = true;
		if (_stream) {
			syncOnExit = false;
			stream = _stream.value();
		} else {
			syncOnExit = true;
			ok &= g_cachedStream.ensure();
			stream = g_cachedStream.stream;
		}
	}
	~StreamHelper() = default;

	cudaStream_t operator()() const { return stream; }

	operator bool() const { return ok; }


	// Sync on stream if not using an existing stream
	bool syncIfSync(const char *msg) {
		if (syncOnExit)
			return sync(msg);
		else
			return ok;
	}
	bool sync(const char *msg) {
		ok &= checkCuda(cudaStreamSynchronize(stream), msg);
		return ok;
	}
private:
	bool ok;
	bool syncOnExit;
	cudaStream_t stream;
};

// A utility class storing a tuple of vectors (of arbitrary types) of a given
// size, to be used as a buffer.
// The cudaEvent_t can be used to synchronize access to the buffer.
template<typename... T>
class TransferConstantsBuffer_t
{
public:
        TransferConstantsBuffer_t(size_t count)
	// Slightly hackish way to construct each vector in the tuple with the count argument.
	// To be able to use the '...' expansion on count, T needs to appear in the expression,
	// so we use a comma operator with void(sizeof(T)) which has no effect.
	: d{ (void(sizeof(T)), count)... }
	{
                checkCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
		          "TransferConstantsBuffer event create");
        }
        ~TransferConstantsBuffer_t() {
                cudaEventDestroy(event);
        }

        std::tuple<std::vector<T>...> d;

        cudaEvent_t event;

};


}

#endif
