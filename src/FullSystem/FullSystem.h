#pragma once
#define MAX_ACTIVE_FRAMES 100

#include <deque>
#include <map>
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "vector"
 
#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/PixelSelector2.h"
#include "system/CameraFrameInput.h"
#include "system/CameraInput.h"
#include "system/LidarProjectionResult.h"
#include "system/RigState.h"
#include "system/VisualState.h"

#include <math.h>

#include <queue>

#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <opencv2/highgui/highgui.hpp>

namespace sdv_loam
{
namespace IOWrap
{
class Output3DWrapper;
}

class PixelSelector;
class PCSyntheticPoint;
class CoarseTracker;
struct FrameHessian;
struct PointHessian;
class CoarseInitializer;
struct ImmaturePointTemporaryResidual;
class ImageAndExposure;
class CoarseDistanceMap;

class EnergyFunctional;

struct RigPoseSnapshot
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	RigPoseSnapshot()
		: timestamp(0.0), cameraId(-1), T_WL(SE3())
	{
	}

	RigPoseSnapshot(double timestamp_, int cameraId_, const SE3& T_WL_)
		: timestamp(timestamp_), cameraId(cameraId_), T_WL(T_WL_)
	{
	}

	double timestamp;
	int cameraId;
	SE3 T_WL;
};

struct CameraCalibration
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	CameraCalibration()
		: T_LC(SE3()),
		  fx(0.0f),
		  cx(0.0f),
		  fy(0.0f),
		  cy(0.0f)
	{
	}

	SE3 T_LC;
	float fx;
	float cx;
	float fy;
	float cy;
};

template<typename T> inline void deleteOut(std::vector<T*> &v, const int i)
{
	delete v[i];
	v[i] = v.back();
	v.pop_back();
}

template<typename T> inline void deleteOutPt(std::vector<T*> &v, const T* i)
{
	delete i;

	for(unsigned int k=0;k<v.size();k++)
		if(v[k] == i)
		{
			v[k] = v.back();
			v.pop_back();
		}
}

template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const int i)
{
	delete v[i];
	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();
}

template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const T* element)
{
	int i=-1;
	for(unsigned int k=0; k<v.size();k++)
	{
		if(v[k] == element)
		{
			i=k;
			break;
		}
	}
	assert(i!=-1);

	for(unsigned int k=i+1; k<v.size();k++)
		v[k-1] = v[k];
	v.pop_back();

	delete element;
}

inline bool eigenTestNan(const MatXX &m, std::string msg)
{
	bool foundNan = false;
	for(int y=0;y<m.rows();y++)
		for(int x=0;x<m.cols();x++)
		{
			if(!std::isfinite((double)m(y,x))) foundNan = true;
		}

	if(foundNan)
	{
		printf("NAN in %s:\n",msg.c_str());
		std::cout << m << "\n\n";
	}


	return foundNan;
}

class FullSystem {
private:
	RigState rigState;
	bool rigPoseValid;
	VisualState visualState;
	std::map<int, VisualState*> visualStatesByCamera;
	std::vector<CameraCalibration, Eigen::aligned_allocator<CameraCalibration>> cameraCalibrations;
	void initializeVisualState(VisualState& visualState);
	void destroyVisualState(VisualState& visualState);
	VisualState& ensureVisualState(int cameraId);

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	FullSystem();
	virtual ~FullSystem();

	// adds a new frame, and creates point & residual structs.
	void addActiveFrame(ImageAndExposure* image, int id);
	void addActiveFrame(const CameraInput& cameraInput);
	void addActiveFrame(const CameraInput& cameraInput, bool updatesRigPose);
	void addActiveFrame(VisualState& visualState, ImageAndExposure* image, int id);
	void addActiveFrame(VisualState& visualState, const CameraInput& cameraInput);
	void addActiveFrame(VisualState& visualState, const CameraInput& cameraInput, bool updatesRigPose);

	// marginalizes a frame. drops / marginalizes points & residuals.
	void marginalizeFrame(FrameHessian* frame);
	void marginalizeFrame(VisualState& visualState, FrameHessian* frame);
	void blockUntilMappingIsFinished();

	float optimize(int mnumOptIts);
	float optimize(VisualState& visualState, int mnumOptIts);

	void printResult(std::string file);

	void debugPlot(std::string name);

	void printFrameLifetimes();
	void printFrameLifetimes(VisualState& visualState);
	// contains pointers to active frames

    std::vector<IOWrap::Output3DWrapper*> outputWrapper;

	bool& isLost;
	bool& initFailed;        			
	bool& initialized;
	bool linearizeOperation;

	void setGammaFunction(float* BInv);

	ros::NodeHandle nh;

	ros::Subscriber subLidarCloud;
    ros::Subscriber subLidarPoseMap;
    ros::Subscriber subLidarPoseOdometer;

    std::queue<double> qTimeLidarPoseMap;
    std::queue<double> qTimeLidarPoseOdometer;

    std::queue<std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>>> qLidarCloud;
    std::map<int, std::queue<cv::Mat> > qImgByCamera;
    std::map<int, std::queue<double> > qTimeImgByCamera;
    std::map<int, std::queue<LidarProjectionResult> > qLidarProjectionResultsByCamera;
    std::map<int, unsigned long long> droppedImagesTooOldByCamera;
    std::map<int, unsigned long long> droppedLidarTooOldByCamera;

    std::queue<Eigen::Matrix3d> qRotationMap;
    std::queue<Eigen::Vector3d> qPositionMap;

    std::queue<Eigen::Matrix3d> qRotationOdometer;
    std::queue<Eigen::Vector3d> qPositionOdometer;

	void initializationValue();
	void loadSensorPrameters(const std::string &pathSensorParameter);
	void loadSensorPrameters(int cameraId, const std::string &pathSensorParameter);
	void enqueueCameraFrame(int cameraId, const cv::Mat& frame, double timestamp);
	void enqueueLidarProjection(const LidarProjectionResult& lidarProjection);
	bool hasQueuedCameraCandidate(int cameraId) const;
	bool synchronizeQueuedCameraCandidate(int cameraId, double maxSyncError);
	std::vector<int> getQueuedCameraCandidateIds() const;
	std::vector<int> getSynchronizedCameraCandidateIds(double maxSyncError);
	CameraFrameInput buildQueuedCameraCandidate(int cameraId) const;
	void popQueuedCameraCandidate(int cameraId);

	float shiTomasiScore(Eigen::Vector3f const *img, int u, int v);

	CalibHessian Hcalib;

	// cam to lidar
	Eigen::Matrix3d Rlc;
	Eigen::Vector3d tlc;

	// cam prameters
	float fx;
	float cx;
	float fy;
	float cy;
	float maxCoarseTrackingRMSE;

	bool addFeaturePoint;
	bool ignoreKF = false;

private:
	void ensureActiveRigCamera();
	void ensureActiveCameraCalibration();
	void syncActiveCameraCalibrationCache();
	void syncActiveRigExtrinsicFromCalibration();
	void syncRigStateFromCameraPose(int cameraId, const SE3& T_WC);
	void syncRigStateFromCameraPose(const SE3& T_WC);
	void syncRigStateFromFrameShell(const FrameShell* shell);

	// opt single point
	int optimizePoint(PointHessian* point, int minObs, bool flagOOB);
	PointHessian* optimizeImmaturePoint(ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);
	PointHessian* optimizeImmaturePoint(VisualState& visualState, ImmaturePoint* point, int minObs, ImmaturePointTemporaryResidual* residuals);

	double linAllPointSinle(PointHessian* point, float outlierTHSlack, bool plot);

	// mainPipelineFunctions
	Vec4 trackNewCoarse(FrameHessian* fh);
	Vec4 trackNewCoarse(VisualState& visualState, FrameHessian* fh);
	void traceNewCoarse(FrameHessian* fh);
	void traceNewCoarse(VisualState& visualState, FrameHessian* fh);
	void activatePoints();
	void activatePointsMT();
	void activatePointsMT(VisualState& visualState);
	void activatePointsOldFirst();
	void flagPointsForRemoval();
	void flagPointsForRemoval(VisualState& visualState);
	void makeNewTraces(VisualState& visualState, FrameHessian* newFrame, float* gtDepth, const LidarProjectionResult& lidarProjection);
	void initializeFromInitializer(FrameHessian* newFrame);
	void initializeFromInitializer(VisualState& visualState, FrameHessian* newFrame);
	void flagFramesForMarginalization(FrameHessian* newFH);
	void flagFramesForMarginalization(VisualState& visualState, FrameHessian* newFH);


	void removeOutliers();
	void removeOutliers(VisualState& visualState);


	
	void setPrecalcValues();   // set precalc values.
	void setPrecalcValues(VisualState& visualState);

	void setMask(cv::Mat &currentFrame, int Ku, int Kv);

	void solveSystem(int iteration, double lambda);  	// solce. eventually migrate to ef.
	void solveSystem(VisualState& visualState, int iteration, double lambda);
	Vec3 linearizeAll(bool fixLinearization);
	Vec3 linearizeAll(VisualState& visualState, bool fixLinearization);
	bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	bool doStepFromBackup(VisualState& visualState, float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
	void backupState(bool backupLastStep);
	void backupState(VisualState& visualState, bool backupLastStep);
	void loadSateBackup();
	void loadSateBackup(VisualState& visualState);
	double calcLEnergy();
	double calcLEnergy(VisualState& visualState);
	double calcMEnergy();
	double calcMEnergy(VisualState& visualState);
	void linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
	void linearizeAll_Reductor(VisualState& visualState, bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid);
	void activatePointsMT_Reductor(VisualState& visualState, std::vector<PointHessian*>* optimized,std::vector<ImmaturePoint*>* toOptimize,int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid);
	void applyRes_Reductor(VisualState& visualState, bool copyJacobians, int min, int max, Vec10* stats, int tid);

	void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);
	void printOptRes(VisualState& visualState, const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

	void debugPlotTracking();

	std::vector<VecX> getNullspaces(
			std::vector<VecX> &nullspaces_pose,
			std::vector<VecX> &nullspaces_scale,
			std::vector<VecX> &nullspaces_affA,
			std::vector<VecX> &nullspaces_affB);
	std::vector<VecX> getNullspaces(
			VisualState& visualState,
			std::vector<VecX> &nullspaces_pose,
			std::vector<VecX> &nullspaces_scale,
			std::vector<VecX> &nullspaces_affA,
			std::vector<VecX> &nullspaces_affB);

	void setNewFrameEnergyTH();
	void setNewFrameEnergyTH(VisualState& visualState);


	void printLogLine();
	void printLogLine(VisualState& visualState);
	void printEvalLine();
	void printEigenValLine();
	void printEigenValLine(VisualState& visualState);
	std::ofstream* calibLog;
	std::ofstream* numsLog;
	std::ofstream* errorsLog;
	std::ofstream* eigenAllLog;
	std::ofstream* eigenPLog;
	std::ofstream* eigenALog;
	std::ofstream* DiagonalLog;
	std::ofstream* variancesLog;
	std::ofstream* nullspacesLog;

	std::ofstream* coarseTrackingLog;

	// statistics
	long int statistics_lastNumOptIts;
	long int statistics_numDroppedPoints;
	long int statistics_numActivatedPoints;
	long int statistics_numCreatedPoints;
	long int statistics_numForceDroppedResBwd;
	long int statistics_numForceDroppedResFwd;
	long int statistics_numMargResFwd;
	long int statistics_numMargResBwd;
	float statistics_lastFineTrackRMSE;

	// =================== changed by tracker-thread. protected by trackMutex ============
	boost::mutex trackMutex;
	std::vector<FrameShell*>& allFrameHistory;
	CoarseInitializer*& coarseInitializer;
	Vec5& lastCoarseRMSE;

	// ================== changed by mapper-thread. protected by mapMutex ===============
	boost::mutex mapMutex;
	std::vector<FrameShell*>& allKeyFramesHistory;

	EnergyFunctional*& ef;
	IndexThreadReduce<Vec10> treadReduce;

	float*& selectionMap;
	float*& selectionMapFromLidar;

	PixelSelector* pixelSelector;
	CoarseDistanceMap*& coarseDistanceMap;

	std::vector<FrameHessian*>& frameHessians;	// ONLY changed in marginalizeFrame and addFrame.
	std::vector<PointFrameResidual*>& activeResiduals;
	float& currentMinActDist;

	std::vector<float>& allResVec;

	// mutex etc. for tracker exchange.
	boost::mutex coarseTrackerSwapMutex;			// if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
	CoarseTracker*& coarseTracker_forNewKF;			// set as as reference. protected by [coarseTrackerSwapMutex].
	CoarseTracker*& coarseTracker;					// always used to track new frames. protected by [trackMutex].
	float minIdJetVisTracker, maxIdJetVisTracker;
	float minIdJetVisDebug, maxIdJetVisDebug;





	// mutex for camToWorl's in shells (these are always in a good configuration).
	boost::mutex shellPoseMutex;
	std::vector<RigPoseSnapshot, Eigen::aligned_allocator<RigPoseSnapshot> > rigPoseSnapshots;

	void makeKeyFrame( FrameHessian* fh);
	void makeNonKeyFrame( FrameHessian* fh);
	void makeKeyFrame(VisualState& visualState, FrameHessian* fh);
	void makeNonKeyFrame(VisualState& visualState, FrameHessian* fh);
	void deliverTrackedFrame(FrameHessian* fh, bool needKF);
	void deliverTrackedFrame(VisualState& visualState, FrameHessian* fh, bool needKF);
	void mappingLoop();
	void mappingLoop(VisualState& visualState);

	// tracking / mapping synchronization. All protected by [trackMapSyncMutex].
	boost::mutex trackMapSyncMutex;
	boost::condition_variable trackedFrameSignal;
	boost::condition_variable mappedFrameSignal;
	std::deque<FrameHessian*> unmappedTrackedFrames;
	int& needNewKFAfter;	// Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.
	boost::thread mappingThread;
	bool runMapping;
	bool needToKetchupMapping;

	int lastRefStopID;

public:
	inline VisualState& getVisualState()
	{
		return visualState;
	}

	VisualState& getVisualState(int cameraId);

	inline const VisualState& getVisualState() const
	{
		return visualState;
	}

	const VisualState& getVisualState(int cameraId) const;

	inline RigState& getRigState()
	{
		return rigState;
	}

	inline const RigState& getRigState() const
	{
		return rigState;
	}

	inline int getActiveCameraId() const
	{
		return rigState.activeCameraId;
	}

	void setActiveCameraId(int cameraId);
	std::vector<int> getConfiguredCameraIds() const;
	int getCameraCalibrationCount() const;
	bool hasCameraCalibration(int cameraId) const;
	CameraCalibration& getCameraCalibration(int cameraId);
	const CameraCalibration& getCameraCalibration(int cameraId) const;
	void setCameraCalibration(int cameraId, const CameraCalibration& calibration);
	void setCameraIntrinsics(int cameraId, float fx, float fy, float cx, float cy);
	void setVisualCalibration(int cameraId, const Eigen::Matrix3f& K);
	CameraCalibration& getActiveCameraCalibration();
	const CameraCalibration& getActiveCameraCalibration() const;

	SE3 getRigPose() const;
	SE3 getRigPoseForCameraPose(int cameraId, const SE3& T_WC) const;
	SE3 getRigPoseForCameraPose(const SE3& T_WC) const;
	SE3 getRigPoseForFrameShell(const FrameShell* shell) const;
	SE3 getActiveCameraPoseFromRig() const;
	void recordRigPoseSnapshot(double timestamp);
	void recordRigPoseSnapshot(double timestamp, int cameraId);
	void recordRigPoseSnapshot(double timestamp, const SE3& T_WL);
	void recordRigPoseSnapshot(double timestamp, int cameraId, const SE3& T_WL);
	bool alignVisualStateToRigPose(int cameraId, const SE3& T_WL);
	bool resetVisualStateForCamera(int cameraId, const SE3& T_WL);
};
}
