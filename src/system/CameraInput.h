#pragma once

#include "system/LidarProjectionResult.h"
#include "util/ImageAndExposure.h"

namespace sdv_loam
{

struct CameraInput
{
	int cameraId;
	ImageAndExposure* image;
	int frameId;
	LidarProjectionResult lidarProjection;
	bool selectionScoreValid;
	double selectionScore;

	CameraInput()
		: cameraId(0),
		  image(0),
		  frameId(0),
		  lidarProjection(),
		  selectionScoreValid(false),
		  selectionScore(0.0)
	{
	}

	CameraInput(int cameraId_, ImageAndExposure* image_, int frameId_)
		: cameraId(cameraId_),
		  image(image_),
		  frameId(frameId_),
		  lidarProjection(),
		  selectionScoreValid(false),
		  selectionScore(0.0)
	{
	}

	CameraInput(int cameraId_, ImageAndExposure* image_, int frameId_, const LidarProjectionResult& lidarProjection_)
		: cameraId(cameraId_),
		  image(image_),
		  frameId(frameId_),
		  lidarProjection(lidarProjection_),
		  selectionScoreValid(false),
		  selectionScore(0.0)
	{
	}

	CameraInput(
		int cameraId_,
		ImageAndExposure* image_,
		int frameId_,
		const LidarProjectionResult& lidarProjection_,
		double selectionScore_)
		: cameraId(cameraId_),
		  image(image_),
		  frameId(frameId_),
		  lidarProjection(lidarProjection_),
		  selectionScoreValid(true),
		  selectionScore(selectionScore_)
	{
	}
};

}
