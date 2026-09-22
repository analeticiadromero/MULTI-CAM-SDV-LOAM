#pragma once

#include <vector>

#include "util/NumType.h"

namespace sdv_loam
{

class CoarseInitializer;
class CoarseTracker;
class CoarseDistanceMap;
class EnergyFunctional;
class FrameShell;
struct FrameHessian;
struct PointFrameResidual;

class VisualState
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	VisualState()
		: isLost(false),
		  initFailed(false),
		  initialized(false),
		  coarseInitializer(0),
		  ef(0),
		  selectionMap(0),
		  selectionMapFromLidar(0),
		  coarseDistanceMap(0),
		  currentMinActDist(0),
		  coarseTracker_forNewKF(0),
		  coarseTracker(0),
		  needNewKFAfter(-1)
	{
		lastCoarseRMSE.setZero();
	}

	bool isLost;
	bool initFailed;
	bool initialized;

	std::vector<FrameShell*> allFrameHistory;
	std::vector<FrameShell*> allKeyFramesHistory;
	Vec5 lastCoarseRMSE;

	CoarseInitializer* coarseInitializer;
	EnergyFunctional* ef;

	float* selectionMap;
	float* selectionMapFromLidar;

	CoarseDistanceMap* coarseDistanceMap;
	std::vector<FrameHessian*> frameHessians;
	std::vector<PointFrameResidual*> activeResiduals;
	float currentMinActDist;

	std::vector<float> allResVec;

	CoarseTracker* coarseTracker_forNewKF;
	CoarseTracker* coarseTracker;

	int needNewKFAfter;
};

}
