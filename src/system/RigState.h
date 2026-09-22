#pragma once

#include <vector>

#include "util/NumType.h"

namespace sdv_loam
{

struct CameraExtrinsic
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	SE3 T_LC;
	SE3 T_CL;

	CameraExtrinsic()
		: T_LC(SE3()), T_CL(SE3())
	{
	}

	explicit CameraExtrinsic(const SE3& T_LC_)
		: T_LC(T_LC_), T_CL(T_LC_.inverse())
	{
	}

	inline void setT_LC(const SE3& T_LC_)
	{
		T_LC = T_LC_;
		T_CL = T_LC.inverse();
	}
};

class RigState
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	RigState()
		: T_WL(SE3()), activeCameraId(0)
	{
	}

	SE3 T_WL;
	std::vector<CameraExtrinsic, Eigen::aligned_allocator<CameraExtrinsic>> cameras;
	int activeCameraId;

	inline bool hasCamera(int cameraId) const
	{
		return cameraId >= 0 && cameraId < static_cast<int>(cameras.size());
	}

	inline const CameraExtrinsic& camera(int cameraId) const
	{
		assert(hasCamera(cameraId));
		return cameras[cameraId];
	}

	inline CameraExtrinsic& camera(int cameraId)
	{
		assert(hasCamera(cameraId));
		return cameras[cameraId];
	}

	inline SE3 getT_WC(int cameraId) const
	{
		return T_WL * camera(cameraId).T_CL;
	}

	inline SE3 getT_CW(int cameraId) const
	{
		return getT_WC(cameraId).inverse();
	}

	inline SE3 getT_WLFromT_WC(int cameraId, const SE3& T_WC) const
	{
		return T_WC * camera(cameraId).T_LC;
	}

	inline void updateT_WLFromCameraPose(int cameraId, const SE3& T_WC)
	{
		T_WL = getT_WLFromT_WC(cameraId, T_WC);
	}
};

}
