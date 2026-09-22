#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"

#include "system/CameraFrameInput.h"
#include "system/VisualState.h"

namespace sdv_loam
{

struct CameraSelectionScore
{
	bool valid;
	double score;
	double syncError;
	double trackingRmse;
	double imageMean;
	double imageStddev;
	double imageGradientMean;
	double imageLaplacianVariance;
	int projectedLidarPoints;
	int activeKeyframes;
	int totalFrames;
	bool initialized;
	bool isLost;
	bool initFailed;

	CameraSelectionScore()
		: valid(false),
		  score(-std::numeric_limits<double>::infinity()),
		  syncError(std::numeric_limits<double>::infinity()),
		  trackingRmse(std::numeric_limits<double>::infinity()),
		  imageMean(0.0),
		  imageStddev(0.0),
		  imageGradientMean(0.0),
		  imageLaplacianVariance(0.0),
		  projectedLidarPoints(0),
		  activeKeyframes(0),
		  totalFrames(0),
		  initialized(false),
		  isLost(false),
		  initFailed(false)
	{
	}
};

inline void computeImageQuality(const cv::Mat& frame, double* mean, double* stddev, double* gradientMean, double* laplacianVariance)
{
	*mean = 0.0;
	*stddev = 0.0;
	*gradientMean = 0.0;
	*laplacianVariance = 0.0;

	if(frame.empty()) return;

	cv::Mat gray;
	if(frame.channels() == 1)
		gray = frame;
	else
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	cv::Scalar cvMean;
	cv::Scalar cvStddev;
	cv::meanStdDev(gray, cvMean, cvStddev);
	*mean = cvMean[0];
	*stddev = cvStddev[0];

	if(gray.cols < 3 || gray.rows < 3) return;

	cv::Mat gradX;
	cv::Mat gradY;
	cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
	cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);

	cv::Mat absGradX;
	cv::Mat absGradY;
	cv::convertScaleAbs(gradX, absGradX);
	cv::convertScaleAbs(gradY, absGradY);

	cv::Mat grad;
	cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0.0, grad);
	*gradientMean = cv::mean(grad)[0];

	cv::Mat laplacian;
	cv::Laplacian(gray, laplacian, CV_64F, 3);
	cv::Scalar lapMean;
	cv::Scalar lapStddev;
	cv::meanStdDev(laplacian, lapMean, lapStddev);
	*laplacianVariance = lapStddev[0] * lapStddev[0];
}

inline CameraSelectionScore scoreCameraCandidate(
	const CameraFrameInput& candidate,
	const VisualState& visualState,
	double maxSyncError,
	double minImageStddev,
	double minImageGradientMean,
	double minImageLaplacianVariance)
{
	CameraSelectionScore result;
	result.syncError = std::fabs(candidate.imageTimestamp - candidate.lidarProjection.lidarTimestamp);
	result.projectedLidarPoints = static_cast<int>(candidate.lidarProjection.cloudPixels.size());
	result.trackingRmse = visualState.lastCoarseRMSE[0];
	result.activeKeyframes = static_cast<int>(visualState.frameHessians.size());
	result.totalFrames = static_cast<int>(visualState.allFrameHistory.size());
	result.initialized = visualState.initialized;
	result.isLost = visualState.isLost;
	result.initFailed = visualState.initFailed;

	if(candidate.frame.empty()) return result;
	if(candidate.cameraId != candidate.lidarProjection.cameraId) return result;
	if(result.syncError > maxSyncError) return result;
	if(result.initFailed) return result;

	computeImageQuality(candidate.frame, &result.imageMean, &result.imageStddev, &result.imageGradientMean, &result.imageLaplacianVariance);
	if(result.imageStddev < minImageStddev) return result;
	if(result.imageGradientMean < minImageGradientMean) return result;
	if(result.imageLaplacianVariance < minImageLaplacianVariance) return result;

	result.valid = true;
	result.score = 0.0;
	result.score += std::min(result.projectedLidarPoints, 3000) * 0.25;
	result.score -= 20000.0 * result.syncError;

	if(result.initialized)
		result.score += 1500.0;
	else
		result.score -= 1000.0;

	result.score += std::min(result.activeKeyframes, 8) * 150.0;
	result.score += std::min(result.totalFrames, 40) * 15.0;
	result.score += std::min(result.imageStddev, 80.0) * 3.0;
	result.score += std::min(result.imageGradientMean, 50.0) * 6.0;
	result.score += std::min(std::log1p(result.imageLaplacianVariance), 10.0) * 60.0;

	if(std::isfinite(result.trackingRmse))
	{
		result.score -= std::min(result.trackingRmse, 100.0) * 180.0;
		if(result.trackingRmse < 4.0)
			result.score += 800.0;
		else if(result.trackingRmse < 8.0)
			result.score += 300.0;
		else if(result.trackingRmse > 12.0)
			result.score -= 1200.0;
	}
	else
	{
		result.score -= 500.0;
	}

	if(result.initialized && result.activeKeyframes < 2)
		result.score -= 600.0;

	if(result.initialized && result.totalFrames < 8)
		result.score -= 300.0;

	if(result.isLost)
		result.score -= 100000.0;

	return result;
}

template <typename VisualStateGetter>
inline bool selectActiveCameraInput(
	const std::vector<CameraFrameInput>& candidates,
	double maxSyncError,
	double minImageStddev,
	double minImageGradientMean,
	double minImageLaplacianVariance,
	int preferredCameraId,
	double switchScoreMargin,
	const CameraFrameInput** selected,
	CameraSelectionScore* selectedScore,
	VisualStateGetter getVisualState)
{
	*selected = 0;
	if(selectedScore != 0)
		*selectedScore = CameraSelectionScore();

	CameraSelectionScore bestScore;
	CameraSelectionScore preferredScore;
	const CameraFrameInput* preferredCandidate = 0;

	for(size_t i = 0; i < candidates.size(); ++i)
	{
		const CameraFrameInput& candidate = candidates[i];
		const CameraSelectionScore score =
			scoreCameraCandidate(candidate, getVisualState(candidate.cameraId), maxSyncError, minImageStddev, minImageGradientMean, minImageLaplacianVariance);

		if(!score.valid) continue;
		if(candidate.cameraId == preferredCameraId)
		{
			preferredCandidate = &candidate;
			preferredScore = score;
		}

		if(*selected == 0 || score.score > bestScore.score)
		{
			*selected = &candidate;
			bestScore = score;
		}
	}

	if(*selected == 0) return false;
	if(preferredCandidate != 0 && (*selected)->cameraId != preferredCameraId && bestScore.score < preferredScore.score + switchScoreMargin)
	{
		*selected = preferredCandidate;
		bestScore = preferredScore;
	}

	if(selectedScore != 0)
		*selectedScore = bestScore;
	return true;
}

}
