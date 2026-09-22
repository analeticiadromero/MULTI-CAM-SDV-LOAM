#pragma once

#include "opencv2/core/core.hpp"

#include "system/LidarProjectionResult.h"

namespace sdv_loam
{

struct CameraFrameInput
{
	int cameraId;
	cv::Mat frame;
	double imageTimestamp;
	LidarProjectionResult lidarProjection;

	CameraFrameInput()
		: cameraId(0),
		  imageTimestamp(0.0),
		  lidarProjection()
	{
	}
};

}
