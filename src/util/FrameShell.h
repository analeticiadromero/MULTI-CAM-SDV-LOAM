#pragma once

#include "util/NumType.h"
#include "algorithm"

namespace sdv_loam
{


class FrameShell
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	int id; 			// INTERNAL ID, starting at zero.
	int cameraId;
	int incoming_id;	// ID passed into DSO
	double timestamp;		// timestamp passed into DSO.

	// set once after tracking
	SE3 camToTrackingRef;
	FrameShell* trackingRef;

	// constantly adapted.
	SE3 camToWorld;				// Legacy visual pose state. Semantically this is T_WC, not T_WL.
	AffLight aff_g2l;
	bool poseValid;
	bool updatesRigPose;

	// statisitcs
	int statistics_outlierResOnThis;
	int statistics_goodResOnThis;
	int marginalizedAt;
	double movedByOpt;

	inline FrameShell()
	{
		id=0;
		cameraId=-1;
		poseValid=true;
		updatesRigPose=true;
		camToWorld = SE3();
		timestamp=0;
		marginalizedAt=-1;
		movedByOpt=0;
		statistics_outlierResOnThis=statistics_goodResOnThis=0;
		trackingRef=0;
		camToTrackingRef = SE3();
	}

	inline const SE3& getT_WC() const
	{
		return camToWorld;
	}

	inline SE3 getT_CW() const
	{
		return camToWorld.inverse();
	}

	inline void setT_WC(const SE3& T_WC)
	{
		camToWorld = T_WC;
	}
};


}
