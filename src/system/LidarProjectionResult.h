#pragma once

#include <vector>

#include <Eigen/StdVector>
#include "Eigen/Core"

namespace sdv_loam
{

struct LidarProjectionResult
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	LidarProjectionResult()
		: cameraId(0),
		  lidarTimestamp(0.0)
	{
	}

	int cameraId;
	double lidarTimestamp;
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> cloudPixels;
};

}
