#include "FullSystem/FullSystem.h"
#include "FullSystem/Reprojector.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iomanip>

namespace sdv_loam
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;

namespace
{
std::string withSuffixBeforeExtension(const std::string& file, const std::string& suffix)
{
	const std::string::size_type slash = file.find_last_of("/\\");
	const std::string::size_type dot = file.find_last_of('.');
	if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
		return file.substr(0, dot) + suffix + file.substr(dot);
	return file + suffix;
}

std::string replaceExtension(const std::string& file, const std::string& extension)
{
	const std::string::size_type slash = file.find_last_of("/\\");
	const std::string::size_type dot = file.find_last_of('.');
	if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
		return file.substr(0, dot) + extension;
	return file + extension;
}

void writeTrajectoryEntry(std::ofstream& kittiFile,
	std::ofstream& timestampFile,
	std::ofstream& tumFile,
	std::ofstream& activeCameraFile,
	double timestamp,
	int cameraId,
	const SE3& T_WL)
{
	Eigen::Quaterniond q;

	q.w() = T_WL.so3().unit_quaternion().w();
	q.x() = T_WL.so3().unit_quaternion().x();
	q.y() = T_WL.so3().unit_quaternion().y();
	q.z() = T_WL.so3().unit_quaternion().z();

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t = T_WL.translation();

	kittiFile << std::fixed << std::setprecision(9)
		<< R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << t(0, 0) << " "
		<< R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << t(1, 0) << " "
		<< R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t(2, 0) << "\n";

	timestampFile << std::fixed << std::setprecision(9) << timestamp << "\n";

	tumFile << std::fixed << std::setprecision(9)
		<< timestamp << " "
		<< t(0, 0) << " " << t(1, 0) << " " << t(2, 0) << " "
		<< q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

	activeCameraFile << std::fixed << std::setprecision(9)
		<< timestamp << " " << cameraId << "\n";
}
}

FullSystem::FullSystem()
	: nh("~"),
	  rigPoseValid(false),
	  isLost(visualState.isLost),
	  initFailed(visualState.initFailed),
	  initialized(visualState.initialized),
	  allFrameHistory(visualState.allFrameHistory),
	  coarseInitializer(visualState.coarseInitializer),
	  lastCoarseRMSE(visualState.lastCoarseRMSE),
	  allKeyFramesHistory(visualState.allKeyFramesHistory),
	  ef(visualState.ef),
	  selectionMap(visualState.selectionMap),
	  selectionMapFromLidar(visualState.selectionMapFromLidar),
	  coarseDistanceMap(visualState.coarseDistanceMap),
	  frameHessians(visualState.frameHessians),
	  activeResiduals(visualState.activeResiduals),
	  currentMinActDist(visualState.currentMinActDist),
	  allResVec(visualState.allResVec),
	  coarseTracker_forNewKF(visualState.coarseTracker_forNewKF),
	  coarseTracker(visualState.coarseTracker),
	  needNewKFAfter(visualState.needNewKFAfter)
{
	maxCoarseTrackingRMSE = 12.0f;
	visualStatesByCamera[0] = &visualState;

	if(rigState.cameras.empty())
	{
		rigState.cameras.emplace_back(SE3());
		rigState.activeCameraId = 0;
	}

	ensureActiveCameraCalibration();
	cameraCalibrations[rigState.activeCameraId].fx = Hcalib.fxl();
	cameraCalibrations[rigState.activeCameraId].fy = Hcalib.fyl();
	cameraCalibrations[rigState.activeCameraId].cx = Hcalib.cxl();
	cameraCalibrations[rigState.activeCameraId].cy = Hcalib.cyl();
	syncActiveRigExtrinsicFromCalibration();
	syncActiveCameraCalibrationCache();

	initializationValue();
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	for(std::map<int, VisualState*>::iterator it = visualStatesByCamera.begin(); it != visualStatesByCamera.end(); ++it)
	{
		destroyVisualState(*it->second);
		if(it->second != &visualState)
			delete it->second;
	}

	delete pixelSelector;
}

void FullSystem::initializeVisualState(VisualState& state)
{
	state.selectionMap = new float[wG[0]*hG[0]];
	state.selectionMapFromLidar = 0;

	state.coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	state.coarseTracker = new CoarseTracker(wG[0], hG[0]);
	state.coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	state.coarseInitializer = new CoarseInitializer(wG[0], hG[0]);

	state.lastCoarseRMSE.setConstant(100);
	state.currentMinActDist = 2;
	state.initialized = false;
	state.isLost = false;
	state.initFailed = false;
	state.needNewKFAfter = -1;

	state.ef = new EnergyFunctional();
	state.ef->red = &this->treadReduce;
}

void FullSystem::destroyVisualState(VisualState& state)
{
	delete[] state.selectionMap;
	state.selectionMap = 0;

	delete[] state.selectionMapFromLidar;
	state.selectionMapFromLidar = 0;

	for(FrameShell* s : state.allFrameHistory)
		delete s;
	state.allFrameHistory.clear();

	delete state.coarseDistanceMap;
	state.coarseDistanceMap = 0;
	delete state.coarseTracker;
	state.coarseTracker = 0;
	delete state.coarseTracker_forNewKF;
	state.coarseTracker_forNewKF = 0;
	delete state.coarseInitializer;
	state.coarseInitializer = 0;
	delete state.ef;
	state.ef = 0;
}

VisualState& FullSystem::ensureVisualState(int cameraId)
{
	assert(cameraId >= 0);

	std::map<int, VisualState*>::iterator it = visualStatesByCamera.find(cameraId);
	if(it != visualStatesByCamera.end())
		return *it->second;

	VisualState* state = new VisualState();
	initializeVisualState(*state);
	visualStatesByCamera[cameraId] = state;
	return *state;
}

void FullSystem::ensureActiveRigCamera()
{
	if(rigState.cameras.empty())
	{
		rigState.cameras.emplace_back(SE3());
	}

	if(rigState.activeCameraId < 0 || rigState.activeCameraId >= static_cast<int>(rigState.cameras.size()))
	{
		rigState.activeCameraId = 0;
	}
}

void FullSystem::ensureActiveCameraCalibration()
{
	ensureActiveRigCamera();

	if(cameraCalibrations.empty())
	{
		cameraCalibrations.emplace_back();
	}

	if(rigState.activeCameraId >= static_cast<int>(cameraCalibrations.size()))
	{
		cameraCalibrations.resize(rigState.activeCameraId + 1);
	}
}

void FullSystem::syncActiveCameraCalibrationCache()
{
	ensureActiveCameraCalibration();
	CameraCalibration& calibration = cameraCalibrations[rigState.activeCameraId];

	Rlc = calibration.T_LC.rotationMatrix();
	tlc = calibration.T_LC.translation();
	fx = calibration.fx;
	cx = calibration.cx;
	fy = calibration.fy;
	cy = calibration.cy;
}

void FullSystem::syncActiveRigExtrinsicFromCalibration()
{
	ensureActiveCameraCalibration();
	rigState.camera(rigState.activeCameraId).setT_LC(cameraCalibrations[rigState.activeCameraId].T_LC);
}

void FullSystem::syncRigStateFromCameraPose(int cameraId, const SE3& T_WC)
{
	if(rigState.hasCamera(cameraId))
		rigState.updateT_WLFromCameraPose(cameraId, T_WC);
	else
		rigState.T_WL = T_WC;

	rigPoseValid = true;
}

void FullSystem::syncRigStateFromCameraPose(const SE3& T_WC)
{
	syncRigStateFromCameraPose(rigState.activeCameraId, T_WC);
}

void FullSystem::syncRigStateFromFrameShell(const FrameShell* shell)
{
	if(shell == 0 || !shell->poseValid) return;
	if(!shell->updatesRigPose) return;
	const int cameraId = shell->cameraId >= 0 ? shell->cameraId : rigState.activeCameraId;
	syncRigStateFromCameraPose(cameraId, shell->getT_WC());
}

SE3 FullSystem::getRigPoseForCameraPose(int cameraId, const SE3& T_WC) const
{
	if(rigState.hasCamera(cameraId))
	{
		return rigState.getT_WLFromT_WC(cameraId, T_WC);
	}

	return T_WC;
}

SE3 FullSystem::getRigPoseForCameraPose(const SE3& T_WC) const
{
	return getRigPoseForCameraPose(rigState.activeCameraId, T_WC);
}

SE3 FullSystem::getRigPose() const
{
	return rigState.T_WL;
}

SE3 FullSystem::getRigPoseForFrameShell(const FrameShell* shell) const
{
	if(shell == 0 || !shell->poseValid) return SE3();
	const int cameraId = shell->cameraId >= 0 ? shell->cameraId : rigState.activeCameraId;
	return getRigPoseForCameraPose(cameraId, shell->getT_WC());
}

SE3 FullSystem::getActiveCameraPoseFromRig() const
{
	if(rigState.hasCamera(rigState.activeCameraId))
	{
		return rigState.getT_WC(rigState.activeCameraId);
	}

	return SE3();
}

void FullSystem::recordRigPoseSnapshot(double timestamp)
{
	recordRigPoseSnapshot(timestamp, rigState.activeCameraId, getRigPose());
}

void FullSystem::recordRigPoseSnapshot(double timestamp, int cameraId)
{
	recordRigPoseSnapshot(timestamp, cameraId, getRigPose());
}

void FullSystem::recordRigPoseSnapshot(double timestamp, const SE3& T_WL)
{
	recordRigPoseSnapshot(timestamp, rigState.activeCameraId, T_WL);
}

void FullSystem::recordRigPoseSnapshot(double timestamp, int cameraId, const SE3& T_WL)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	rigPoseSnapshots.push_back(RigPoseSnapshot(timestamp, cameraId, T_WL));
}

bool FullSystem::alignVisualStateToRigPose(int cameraId, const SE3& T_WL)
{
	if(!rigState.hasCamera(cameraId))
		return false;

	std::map<int, VisualState*>::iterator stateIt = visualStatesByCamera.find(cameraId);
	if(stateIt == visualStatesByCamera.end() || stateIt->second == 0)
		return false;

	VisualState& state = *stateIt->second;
	FrameShell* referenceShell = 0;
	for(std::vector<FrameShell*>::reverse_iterator it = state.allFrameHistory.rbegin();
		it != state.allFrameHistory.rend();
		++it)
	{
		if(*it != 0 && (*it)->poseValid)
		{
			referenceShell = *it;
			break;
		}
	}

	if(referenceShell == 0)
		return false;

	const SE3 targetReferenceT_WC = T_WL * rigState.camera(cameraId).T_CL;
	const SE3 alignment = targetReferenceT_WC * referenceShell->getT_WC().inverse();

	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
	for(FrameShell* shell : state.allFrameHistory)
	{
		if(shell == 0 || !shell->poseValid)
			continue;
		shell->setT_WC(alignment * shell->getT_WC());
	}

	for(FrameHessian* fh : state.frameHessians)
	{
		if(fh == 0 || fh->shell == 0 || !fh->shell->poseValid)
			continue;
		fh->setT_WC_scaled(fh->shell->getT_WC(), fh->shell->aff_g2l);
	}

	rigState.T_WL = T_WL;
	rigPoseValid = true;
	return true;
}

bool FullSystem::resetVisualStateForCamera(int cameraId, const SE3& T_WL)
{
	if(cameraId < 0)
		return false;

	// Do not destroy the previous state here. Mapping/tracking can still hold
	// FrameHessian/residual pointers from it while a camera switch is processed.
	// Replacing the camera slot with a fresh state avoids dangling pointers and
	// lets the old state die with the process after this experimental run.
	VisualState* state = new VisualState();
	initializeVisualState(*state);
	visualStatesByCamera[cameraId] = state;

	rigState.T_WL = T_WL;
	rigPoseValid = true;
	return true;
}

CameraCalibration& FullSystem::getActiveCameraCalibration()
{
	ensureActiveCameraCalibration();
	return cameraCalibrations[rigState.activeCameraId];
}

const CameraCalibration& FullSystem::getActiveCameraCalibration() const
{
	assert(rigState.activeCameraId >= 0 && rigState.activeCameraId < static_cast<int>(cameraCalibrations.size()));
	return cameraCalibrations[rigState.activeCameraId];
}

void FullSystem::setActiveCameraId(int cameraId)
{
	assert(cameraId >= 0);

	if(cameraId >= static_cast<int>(rigState.cameras.size()))
	{
		rigState.cameras.resize(cameraId + 1);
	}
	if(cameraId >= static_cast<int>(cameraCalibrations.size()))
	{
		cameraCalibrations.resize(cameraId + 1);
	}

	rigState.activeCameraId = cameraId;
	ensureVisualState(cameraId);
	syncActiveRigExtrinsicFromCalibration();
	syncActiveCameraCalibrationCache();
}

VisualState& FullSystem::getVisualState(int cameraId)
{
	return ensureVisualState(cameraId);
}

const VisualState& FullSystem::getVisualState(int cameraId) const
{
	std::map<int, VisualState*>::const_iterator it = visualStatesByCamera.find(cameraId);
	if(it != visualStatesByCamera.end())
		return *it->second;

	return visualState;
}

std::vector<int> FullSystem::getConfiguredCameraIds() const
{
	std::vector<int> cameraIds;
	cameraIds.reserve(cameraCalibrations.size());
	for(size_t cameraId = 0; cameraId < cameraCalibrations.size(); ++cameraId)
		cameraIds.push_back(static_cast<int>(cameraId));

	return cameraIds;
}

void FullSystem::enqueueCameraFrame(int cameraId, const cv::Mat& frame, double timestamp)
{
	assert(cameraId >= 0);
	qImgByCamera[cameraId].push(frame);
	qTimeImgByCamera[cameraId].push(timestamp);
}

void FullSystem::enqueueLidarProjection(const LidarProjectionResult& lidarProjection)
{
	assert(lidarProjection.cameraId >= 0);
	qLidarProjectionResultsByCamera[lidarProjection.cameraId].push(lidarProjection);
}

bool FullSystem::hasQueuedCameraCandidate(int cameraId) const
{
	std::map<int, std::queue<cv::Mat> >::const_iterator imageIt = qImgByCamera.find(cameraId);
	std::map<int, std::queue<double> >::const_iterator timeIt = qTimeImgByCamera.find(cameraId);
	std::map<int, std::queue<LidarProjectionResult> >::const_iterator lidarIt =
		qLidarProjectionResultsByCamera.find(cameraId);

	return imageIt != qImgByCamera.end() &&
		timeIt != qTimeImgByCamera.end() &&
		lidarIt != qLidarProjectionResultsByCamera.end() &&
		!imageIt->second.empty() &&
		!timeIt->second.empty() &&
		!lidarIt->second.empty();
}

bool FullSystem::synchronizeQueuedCameraCandidate(int cameraId, double maxSyncError)
{
	std::map<int, std::queue<cv::Mat> >::iterator imageIt = qImgByCamera.find(cameraId);
	std::map<int, std::queue<double> >::iterator timeIt = qTimeImgByCamera.find(cameraId);
	std::map<int, std::queue<LidarProjectionResult> >::iterator lidarIt =
		qLidarProjectionResultsByCamera.find(cameraId);

	if(imageIt == qImgByCamera.end() ||
		timeIt == qTimeImgByCamera.end() ||
		lidarIt == qLidarProjectionResultsByCamera.end())
	{
		return false;
	}

	while(!imageIt->second.empty() && !timeIt->second.empty() && !lidarIt->second.empty())
	{
		const double imageTimestamp = timeIt->second.front();
		const double lidarTimestamp = lidarIt->second.front().lidarTimestamp;
		const double syncError = fabs(imageTimestamp - lidarTimestamp);

		if(syncError <= maxSyncError)
			return true;

		if(imageTimestamp < lidarTimestamp)
		{
			imageIt->second.pop();
			timeIt->second.pop();
			droppedImagesTooOldByCamera[cameraId]++;
		}
		else
		{
			lidarIt->second.pop();
			droppedLidarTooOldByCamera[cameraId]++;
		}
	}

	return false;
}

std::vector<int> FullSystem::getQueuedCameraCandidateIds() const
{
	std::vector<int> cameraIds;
	for(std::map<int, std::queue<cv::Mat> >::const_iterator it = qImgByCamera.begin();
		it != qImgByCamera.end();
		++it)
	{
		if(hasQueuedCameraCandidate(it->first))
			cameraIds.push_back(it->first);
	}

	return cameraIds;
}

std::vector<int> FullSystem::getSynchronizedCameraCandidateIds(double maxSyncError)
{
	std::vector<int> cameraIds;
	for(std::map<int, std::queue<cv::Mat> >::iterator it = qImgByCamera.begin();
		it != qImgByCamera.end();
		++it)
	{
		if(synchronizeQueuedCameraCandidate(it->first, maxSyncError))
			cameraIds.push_back(it->first);
	}

	return cameraIds;
}

CameraFrameInput FullSystem::buildQueuedCameraCandidate(int cameraId) const
{
	assert(hasQueuedCameraCandidate(cameraId));

	CameraFrameInput input;
	input.cameraId = cameraId;
	input.frame = qImgByCamera.find(cameraId)->second.front();
	input.imageTimestamp = qTimeImgByCamera.find(cameraId)->second.front();
	input.lidarProjection = qLidarProjectionResultsByCamera.find(cameraId)->second.front();
	return input;
}

void FullSystem::popQueuedCameraCandidate(int cameraId)
{
	assert(hasQueuedCameraCandidate(cameraId));

	qImgByCamera[cameraId].pop();
	qTimeImgByCamera[cameraId].pop();
	qLidarProjectionResultsByCamera[cameraId].pop();
}

int FullSystem::getCameraCalibrationCount() const
{
	return static_cast<int>(cameraCalibrations.size());
}

bool FullSystem::hasCameraCalibration(int cameraId) const
{
	return cameraId >= 0 && cameraId < static_cast<int>(cameraCalibrations.size());
}

CameraCalibration& FullSystem::getCameraCalibration(int cameraId)
{
	assert(cameraId >= 0);

	ensureActiveRigCamera();
	if(cameraId >= static_cast<int>(cameraCalibrations.size()))
	{
		cameraCalibrations.resize(cameraId + 1);
	}
	if(cameraId >= static_cast<int>(rigState.cameras.size()))
	{
		rigState.cameras.resize(cameraId + 1);
	}

	return cameraCalibrations[cameraId];
}

const CameraCalibration& FullSystem::getCameraCalibration(int cameraId) const
{
	assert(hasCameraCalibration(cameraId));
	return cameraCalibrations[cameraId];
}

void FullSystem::loadSensorPrameters(const std::string &pathSensorParameter)
{
	loadSensorPrameters(rigState.activeCameraId, pathSensorParameter);
}

void FullSystem::setCameraCalibration(int cameraId, const CameraCalibration& calibration)
{
	CameraCalibration& targetCalibration = getCameraCalibration(cameraId);
	targetCalibration = calibration;
	rigState.camera(cameraId).setT_LC(targetCalibration.T_LC);

	if(cameraId == rigState.activeCameraId)
		syncActiveCameraCalibrationCache();
}

void FullSystem::setCameraIntrinsics(int cameraId, float fx_, float fy_, float cx_, float cy_)
{
	CameraCalibration& calibration = getCameraCalibration(cameraId);
	calibration.fx = fx_;
	calibration.fy = fy_;
	calibration.cx = cx_;
	calibration.cy = cy_;

	if(cameraId == rigState.activeCameraId)
		syncActiveCameraCalibrationCache();
}

void FullSystem::setVisualCalibration(int cameraId, const Eigen::Matrix3f& K)
{
	setActiveCameraId(cameraId);

	VecC initial_value = VecC::Zero();
	initial_value[0] = K(0, 0);
	initial_value[1] = K(1, 1);
	initial_value[2] = K(0, 2);
	initial_value[3] = K(1, 2);

	Hcalib.setValueScaled(initial_value);
	Hcalib.value_zero = Hcalib.value;
	Hcalib.value_minus_value_zero.setZero();
	Hcalib.step.setZero();
	Hcalib.step_backup.setZero();
	Hcalib.value_backup = Hcalib.value;

	setCameraIntrinsics(cameraId, K(0, 0), K(1, 1), K(0, 2), K(1, 2));
}

void FullSystem::loadSensorPrameters(int cameraId, const std::string &pathSensorParameter)
{
	std::ifstream infile;
    infile.open(pathSensorParameter.c_str());

    int numLine = 1;
	CameraCalibration calibration = getCameraCalibration(cameraId);
	calibration.fx = Hcalib.fxl();
	calibration.fy = Hcalib.fyl();
	calibration.cx = Hcalib.cxl();
	calibration.cy = Hcalib.cyl();
	Eigen::Matrix3d Rlc;
	Eigen::Vector3d tlc;

    while(!infile.eof())
    {
        std::string s;
        getline(infile,s);
        if(!s.empty() && numLine == 1)
        {
            numLine++;
        }
        else if(!s.empty() && numLine == 2)
        {
        	std::stringstream ss;
            ss << s;
            ss >> Rlc(0, 0); ss >> Rlc(0, 1); ss >> Rlc(0, 2); ss >> tlc(0, 0);
            numLine++;
        }
        else if(!s.empty() && numLine == 3)
        {
        	std::stringstream ss;
            ss << s;
            ss >> Rlc(1, 0); ss >> Rlc(1, 1); ss >> Rlc(1, 2); ss >> tlc(1, 0);
            numLine++;
        }
        else if(!s.empty() && numLine == 4)
        {
        	std::stringstream ss;
            ss << s;
            ss >> Rlc(2, 0); ss >> Rlc(2, 1); ss >> Rlc(2, 2); ss >> tlc(2, 0);
            calibration.T_LC = SE3(Rlc, tlc);
            setCameraCalibration(cameraId, calibration);
            numLine++;
        }
    }

    infile.close();
}

void FullSystem::initializationValue()
{
	int retstat =0;
	if(setting_logStuff)
	{
		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);

		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);

	initializeVisualState(visualState);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(static_cast<void (FullSystem::*)()>(&FullSystem::mappingLoop), this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);

	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::vector<FrameShell*> validShells;
	if(rigPoseSnapshots.empty())
	{
		std::vector<FrameShell*> shells;
		for(std::map<int, VisualState*>::const_iterator it = visualStatesByCamera.begin();
			it != visualStatesByCamera.end();
			++it)
		{
			for(FrameShell* s : it->second->allFrameHistory)
				shells.push_back(s);
		}

		std::sort(shells.begin(), shells.end(), [](const FrameShell* a, const FrameShell* b) {
			return a->timestamp < b->timestamp;
		});

		for(FrameShell* s : shells)
		{
			if(!s->poseValid) continue;
			if(!s->updatesRigPose) continue;
			validShells.push_back(s);
		}

		if(validShells.empty())
			return;
	}

	const std::string timestampFile = withSuffixBeforeExtension(file, "_timestamps");
	const std::string activeCameraFile = withSuffixBeforeExtension(file, "_active_camera");
	const std::string tumFile = replaceExtension(file, ".tum");
	std::ofstream myfile;
	std::ofstream timefile;
	std::ofstream camerafile;
	std::ofstream tumfile;
	myfile.open(file.c_str());
	timefile.open(timestampFile.c_str());
	camerafile.open(activeCameraFile.c_str());
	tumfile.open(tumFile.c_str());

	if(!rigPoseSnapshots.empty())
	{
		for(const RigPoseSnapshot& snapshot : rigPoseSnapshots)
		{
			writeTrajectoryEntry(myfile, timefile, tumfile, camerafile, snapshot.timestamp, snapshot.cameraId, snapshot.T_WL);
		}
		myfile.close();
		timefile.close();
		camerafile.close();
		tumfile.close();
		return;
	}

	for(FrameShell* s : validShells)
	{
		const SE3 T_WL = getRigPoseForFrameShell(s);
		writeTrajectoryEntry(myfile, timefile, tumfile, camerafile, s->timestamp, s->cameraId, T_WL);
	}
	myfile.close();
	timefile.close();
	camerafile.close();
	tumfile.close();
}

Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{
	return trackNewCoarse(visualState, fh);
}

Vec4 FullSystem::trackNewCoarse(VisualState& visualState, FrameHessian* fh)
{

	assert(visualState.allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	FrameHessian* lastF = visualState.coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);
	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

	if(visualState.allFrameHistory.size() == 2) {
		initializeFromInitializer(visualState, fh);

		lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() ));

		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
        {
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        }
		
		visualState.coarseTracker->makeK(&Hcalib);
		visualState.coarseTracker->setCTRefForFirstFrame(visualState.frameHessians);

		lastF = visualState.coarseTracker->lastRef;
	}
	else
	{
		FrameShell* slast = visualState.allFrameHistory[visualState.allFrameHistory.size()-2];
		FrameShell* sprelast = visualState.allFrameHistory[visualState.allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	
			// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.

		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);

	//! as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	//! I'll keep track of the so-far best achieved residual for each level in achievedRes. 
	//! If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.	 

	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;

	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		
			bool trackingIsGood = visualState.coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0 && !setting_debugout_runquiet)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					visualState.coarseTracker->lastResiduals[0],
					visualState.coarseTracker->lastResiduals[1],
					visualState.coarseTracker->lastResiduals[2],
					visualState.coarseTracker->lastResiduals[3],
					visualState.coarseTracker->lastResiduals[4]);
		}

		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)visualState.coarseTracker->lastResiduals[0]) && !(visualState.coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			flowVecs = visualState.coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
					if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > visualState.coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
						achievedRes[i] = visualState.coarseTracker->lastResiduals[i];
			}
		}

        if(haveOneGood &&  achievedRes[0] < visualState.lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
		if(!setting_debugout_runquiet)
			printf("BIG ERROR! tracking failed entirely. Reject frame.\n");
		return Vec4(NAN, NAN, NAN, NAN);
	}

	visualState.lastCoarseRMSE = achievedRes;

	if(!std::isfinite((float)achievedRes[0]) || achievedRes[0] > maxCoarseTrackingRMSE)
	{
		if(!setting_debugout_runquiet)
			printf("Coarse tracking rejected: RMSE %.3f > %.3f.\n", achievedRes[0], maxCoarseTrackingRMSE);
		return Vec4(NAN, NAN, NAN, NAN);
	}

	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	// Legacy tracking still propagates the camera pose T_WC through the visual chain.
	fh->shell->setT_WC(fh->shell->trackingRef->getT_WC() * fh->shell->camToTrackingRef);

	std::vector<std::pair<PointHessian*, Eigen::Vector2d> > overlap_pts;
	Reprojector reprojector_ = Reprojector(&Hcalib, fh, visualState.frameHessians);
	reprojector_.reprojectMap(fh, overlap_pts);

	SE3 curToWorld = fh->shell->getT_WC();
	visualState.coarseTracker->structPoseEstimation(curToWorld, overlap_pts);
	fh->shell->setT_WC(curToWorld);
	syncRigStateFromFrameShell(fh->shell);

	SE3 Twl = fh->shell->trackingRef->getT_WC();
	fh->shell->camToTrackingRef = Twl.inverse() * curToWorld;

	if(visualState.coarseTracker->firstCoarseRMSE < 0)
		visualState.coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	traceNewCoarse(visualState, fh);
}

void FullSystem::traceNewCoarse(VisualState& visualState, FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : visualState.frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
}


//@ 处理挑选出来待激活的点
void FullSystem::activatePointsMT_Reductor(
		VisualState& visualState,
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[visualState.frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint(visualState, (*toOptimize)[k],1,tr);
	}
	delete[] tr;
}

void FullSystem::activatePointsMT()
{
	activatePointsMT(visualState);
}

void FullSystem::activatePointsMT(VisualState& visualState)
{
	if(visualState.ef->nPoints < setting_desiredPointDensity*0.66)
		visualState.currentMinActDist -= 0.8;
	if(visualState.ef->nPoints < setting_desiredPointDensity*0.8)
		visualState.currentMinActDist -= 0.5;
	else if(visualState.ef->nPoints < setting_desiredPointDensity*0.9)
		visualState.currentMinActDist -= 0.2;
	else if(visualState.ef->nPoints < setting_desiredPointDensity)
		visualState.currentMinActDist -= 0.1;

	if(visualState.ef->nPoints > setting_desiredPointDensity*1.5)
		visualState.currentMinActDist += 0.8;
	if(visualState.ef->nPoints > setting_desiredPointDensity*1.3)
		visualState.currentMinActDist += 0.5;
	if(visualState.ef->nPoints > setting_desiredPointDensity*1.15)
		visualState.currentMinActDist += 0.2;
	if(visualState.ef->nPoints > setting_desiredPointDensity)
		visualState.currentMinActDist += 0.1;

	if(visualState.currentMinActDist < 0) visualState.currentMinActDist = 0;
	if(visualState.currentMinActDist > 4) visualState.currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                visualState.currentMinActDist, (int)(setting_desiredPointDensity), visualState.ef->nPoints);



	FrameHessian* newestHs = visualState.frameHessians.back();

	// make dist map.
	visualState.coarseDistanceMap->makeK(&Hcalib);
	visualState.coarseDistanceMap->makeDistanceMap(visualState.frameHessians, newestHs);

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);

	for(FrameHessian* host : visualState.frameHessians)		// go through all active frames
	{
		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (visualState.coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * visualState.coarseDistanceMap->Ki[0]);
		Vec3f Kt = (visualState.coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			if(ph->isFromSensor == false && host == newestHs)
				continue;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}
			
			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;

			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
					delete ph;
					host->immaturePoints[i]=0;
				}
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{
				float dist = visualState.coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=visualState.currentMinActDist* ph->my_type)
				{
					visualState.coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, boost::ref(visualState), &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
	{
		activatePointsMT_Reductor(visualState, &optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
	}

	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			visualState.ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				visualState.ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			delete ph;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

	for(FrameHessian* host : visualState.frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}

void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	flagPointsForRemoval(visualState);
}

void FullSystem::flagPointsForRemoval(VisualState& visualState)
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	{
		for(int i=((int)visualState.frameHessians.size())-1;i>=0 && i >= ((int)visualState.frameHessians.size());i--)
			if(!visualState.frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(visualState.frameHessians[i]);

		for(int i=0; i< (int)visualState.frameHessians.size();i++)
			if(visualState.frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(visualState.frameHessians[i]);
	}

	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : visualState.frameHessians)
	{
		if(host == visualState.frameHessians.back()) continue;

		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;

				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					float Hdd_acc = 0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);

						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(visualState.ef);
							ngoodRes++;
						}
					}

                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				}

				host->pointHessians[i]=0;
			}
		}

		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}
}

void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{
	addActiveFrame(CameraInput(rigState.activeCameraId, image, id));
}

void FullSystem::addActiveFrame(const CameraInput& cameraInput)
{
	addActiveFrame(cameraInput, true);
}

void FullSystem::addActiveFrame(const CameraInput& cameraInput, bool updatesRigPose)
{
	addActiveFrame(ensureVisualState(cameraInput.cameraId), cameraInput, updatesRigPose);
}

void FullSystem::addActiveFrame(VisualState& visualState, ImageAndExposure* image, int id )
{
	addActiveFrame(visualState, CameraInput(rigState.activeCameraId, image, id));
}

void FullSystem::addActiveFrame(VisualState& visualState, const CameraInput& cameraInput)
{
	addActiveFrame(visualState, cameraInput, true);
}

void FullSystem::addActiveFrame(VisualState& visualState, const CameraInput& cameraInput, bool updatesRigPose)
{
	setActiveCameraId(cameraInput.cameraId);

	ImageAndExposure* image = cameraInput.image;
	int id = cameraInput.frameId;

    if(visualState.isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);

	FrameHessian* fh = new FrameHessian();
	fh->cameraId = cameraInput.cameraId;
	FrameShell* shell = new FrameShell();
	shell->cameraId = cameraInput.cameraId;
	shell->updatesRigPose = updatesRigPose;
	const bool anchorShellToRig =
		updatesRigPose &&
		rigPoseValid &&
		rigState.hasCamera(cameraInput.cameraId);
	shell->setT_WC(anchorShellToRig ? rigState.getT_WC(cameraInput.cameraId) : SE3());
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = visualState.allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;
	fh->lidarProjection = cameraInput.lidarProjection;
	visualState.allFrameHistory.push_back(shell);

	std::cout << std::fixed << "current time = " << shell->timestamp
			  << " camera = " << cameraInput.cameraId
			  << (updatesRigPose ? "" : " warmup")
			  << " score = ";
	if(cameraInput.selectionScoreValid)
		std::cout << cameraInput.selectionScore;
	else
		std::cout << "nan";
	std::cout << " lidar_points = " << cameraInput.lidarProjection.cloudPixels.size()
			  << " sync_error = " << fabs(shell->timestamp - cameraInput.lidarProjection.lidarTimestamp)
			  << std::endl;

	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);

	if(!visualState.initialized)
	{
		if(visualState.coarseInitializer->frameID < 0)
		{
			if(!visualState.coarseInitializer->setFirstFromLidar(&Hcalib, fh, this, fh->lidarProjection))
			{
				visualState.allFrameHistory.pop_back();
				delete shell;
				fh->shell = 0;
				delete fh;
				return;
			}
			visualState.initialized = true;
		}
		return;
	}
	else	// do front-end operation.
	{
		if(visualState.coarseTracker_forNewKF->refFrameID > visualState.coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = visualState.coarseTracker;
			visualState.coarseTracker=visualState.coarseTracker_forNewKF; 
			visualState.coarseTracker_forNewKF=tmp;
		}

		Vec4 tres = trackNewCoarse(visualState, fh);
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            if(!setting_debugout_runquiet)
                printf("Tracking rejected frame %d.\n", id);
			visualState.allFrameHistory.pop_back();
			delete shell;
			fh->shell = 0;
			delete fh;
            return;
        }

		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = visualState.allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - visualState.allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(visualState.coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					visualState.coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = visualState.allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +  
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) + 	
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +	
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2*visualState.coarseTracker->firstCoarseRMSE < tres[0];

		}

		if(ignoreKF && fh->shell->timestamp - visualState.allKeyFramesHistory.back()->timestamp <= 0.15)
			needToMakeKF = false;

        if(fh->shell->updatesRigPose)
        {
            for(IOWrap::Output3DWrapper* ow : outputWrapper)
                ow->publishCamPose(fh->shell, &Hcalib);
        }

		lock.unlock();
		deliverTrackedFrame(visualState, fh, needToMakeKF);
		return;
	}
}

void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{
	deliverTrackedFrame(visualState, fh, needKF);
}

void FullSystem::deliverTrackedFrame(VisualState& visualState, FrameHessian* fh, bool needKF)
{

	//! 顺序执行
	if(linearizeOperation) 
	{
		if(goStepByStep && lastRefStopID != visualState.coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = visualState.coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(visualState, fh);
		else makeNonKeyFrame(visualState, fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) visualState.needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(visualState.coarseTracker_forNewKF->refFrameID == -1 && visualState.coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	mappingLoop(visualState);
}

void FullSystem::mappingLoop(VisualState& defaultVisualState)
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	(void)defaultVisualState;

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();
		VisualState& visualState = getVisualState(fh->cameraId);


		// guaranteed to make a KF for the very first two tracked frames.
		if(visualState.allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(visualState, fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(visualState, fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->setT_WC(fh->shell->trackingRef->getT_WC() * fh->shell->camToTrackingRef);
					fh->setT_WC_scaled(fh->shell->getT_WC(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || visualState.needNewKFAfter >= visualState.frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(visualState, fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(visualState, fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	makeNonKeyFrame(visualState, fh);
}

void FullSystem::makeNonKeyFrame(VisualState& visualState, FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);

		fh->shell->setT_WC(fh->shell->trackingRef->getT_WC() * fh->shell->camToTrackingRef);
		fh->setT_WC_scaled(fh->shell->getT_WC(),fh->shell->aff_g2l);
	}

	traceNewCoarse(visualState, fh);
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	makeKeyFrame(visualState, fh);
}

void FullSystem::makeKeyFrame(VisualState& visualState, FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->setT_WC(fh->shell->trackingRef->getT_WC() * fh->shell->camToTrackingRef);
		fh->setT_WC_scaled(fh->shell->getT_WC(),fh->shell->aff_g2l);
	}

	traceNewCoarse(visualState, fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	flagFramesForMarginalization(visualState, fh);

	if(visualState.allKeyFramesHistory.size() >= 2)
	{
		float timeLast = visualState.allKeyFramesHistory.back()->timestamp;
		float timeLastLast = visualState.allKeyFramesHistory[visualState.allKeyFramesHistory.size() - 2]->timestamp;
		Eigen::Vector3d tLast = visualState.allKeyFramesHistory.back()->getT_WC().translation();
		Eigen::Vector3d tLastLast = visualState.allKeyFramesHistory[visualState.allKeyFramesHistory.size() - 2]->getT_WC().translation();
		float distance = sqrt((tLast[0] - tLastLast[0]) * (tLast[0] - tLastLast[0]) + (tLast[1] - tLastLast[1]) * (tLast[1] - tLastLast[1]) + 
			(tLast[2] - tLastLast[2]) * (tLast[2] - tLastLast[2]));
		float speed = distance / (timeLast - timeLastLast);

		if(speed < 10)
			ignoreKF = true;
		else
			ignoreKF = false;
	}

	fh->idx = visualState.frameHessians.size();
	visualState.frameHessians.push_back(fh);
	fh->frameID = visualState.allKeyFramesHistory.size();
	visualState.allKeyFramesHistory.push_back(fh->shell);
	visualState.ef->insertFrame(fh, &Hcalib);

	setPrecalcValues(visualState);

	makeNewTraces(visualState, fh, 0, fh->lidarProjection);
	for(ImmaturePoint* ph : fh->immaturePoints)
		if(ph->isFromSensor == true){
			ph->lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}

	int numFwdResAdde=0;
	for(FrameHessian* fh1 : visualState.frameHessians)
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			visualState.ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}

	activatePointsMT(visualState);
	visualState.ef->makeIDX();

	for(int i = 0; i < visualState.frameHessians.size() - 1; i++)
	{
		std::vector<std::pair<PointHessian*, Eigen::Vector2d> > overlap_pts;
		Reprojector reprojector_ = Reprojector(&Hcalib, fh, visualState.frameHessians);
		reprojector_.backprojectMap(fh,visualState.frameHessians[i], overlap_pts);
	}

	for(int i = visualState.frameHessians.size() - 2; i >= 0; i--)
	{
		std::vector<std::pair<PointHessian*, Eigen::Vector2d> > overlap_pts;
		Reprojector reprojector_ = Reprojector(&Hcalib, visualState.frameHessians[i], visualState.frameHessians);
		reprojector_.backprojectMap(visualState.frameHessians[i],fh, overlap_pts);
	}

	for(FrameHessian* pf : visualState.frameHessians){
		int numMatch = 0;
		for(PointHessian* pt : pf->pointHessians)
			for(PointFrameResidual* pr : pt->residuals)
			{
				if(!pr->hasMatcher)
				{
					pr->findMatches();
				}
				if(pr->hasMatcher)
					numMatch++;
			}
	}

	fh->frameEnergyTH = visualState.frameHessians.back()->frameEnergyTH;
	float rmse = optimize(visualState, setting_maxOptIterations);

    if(visualState.isLost) return;

	removeOutliers(visualState);

	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);

		visualState.coarseTracker_forNewKF->makeK(&Hcalib);
		visualState.coarseTracker_forNewKF->setCoarseTrackingRef(visualState.frameHessians);

        if(!setting_debugout_runquiet && !outputWrapper.empty())
        {
            visualState.coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
            visualState.coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
        }
	}

	if(!setting_debugout_runquiet)
		debugPlot("post Optimize");

	flagPointsForRemoval(visualState);
	visualState.ef->dropPointsF();

	getNullspaces(
			visualState,
			visualState.ef->lastNullspaces_pose,
			visualState.ef->lastNullspaces_scale,
			visualState.ef->lastNullspaces_affA,
			visualState.ef->lastNullspaces_affB);

	visualState.ef->marginalizePointsF();

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(visualState.ef->connectivityMap);
        ow->publishKeyframes(visualState.frameHessians, false, &Hcalib);
    }

	for(unsigned int i=0;i<visualState.frameHessians.size();i++)
		if(visualState.frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(visualState, visualState.frameHessians[i]); i=0;}

	printLogLine(visualState);
}

void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	initializeFromInitializer(visualState, newFrame);
}

void FullSystem::initializeFromInitializer(VisualState& visualState, FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	FrameHessian* firstFrame = visualState.coarseInitializer->firstFrame;
	firstFrame->idx = visualState.frameHessians.size();
	visualState.frameHessians.push_back(firstFrame);
	firstFrame->frameID = visualState.allKeyFramesHistory.size();
	visualState.allKeyFramesHistory.push_back(firstFrame->shell);
	visualState.ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues(visualState);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<visualState.coarseInitializer->numPoints[0];i++)
	{
		sumID += visualState.coarseInitializer->points[0][i].iR;
		numID++;
	}

	float keepPercentage = setting_desiredPointDensity / visualState.coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), visualState.coarseInitializer->numPoints[0] );

	for(int i=0;i<visualState.coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue;

		Pnt* point = visualState.coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(point->isFromSensor == true)
		{
			pt->idepth_min = 1 / point->mdepth;
        	pt->idepth_max = 1 / point->mdepth;
		}

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }

		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->midepth);
		ph->setIdepthZero(point->midepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		if(point->isFromSensor == true)
			ph->isFromSensor = true;
		else
			ph->isFromSensor = false;

		firstFrame->pointHessians.push_back(ph);
		visualState.ef->insertPoint(ph);
	}

	SE3 firstToNew = visualState.coarseInitializer->thisToNext;
	const bool alignInitializerToRig = newFrame->shell->updatesRigPose && rigPoseValid && rigState.hasCamera(newFrame->cameraId);
	const SE3 newFrameT_WC = alignInitializerToRig ? rigState.getT_WC(newFrame->cameraId) : firstToNew.inverse();
	const SE3 firstFrameT_WC = alignInitializerToRig ? newFrameT_WC * firstToNew : SE3();

	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->setT_WC(firstFrameT_WC);
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->setT_WC(newFrameT_WC);
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	visualState.initialized=true;
	syncRigStateFromFrameShell(newFrame->shell);
	if(alignInitializerToRig && !setting_debugout_runquiet)
		printf("aligned camera %d initializer to rig pose.\n", newFrame->cameraId);
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::setMask(cv::Mat &currentFrame, int Ku, int Kv)
{
	for(int i = Ku - pixelSelector->currentPotential; i <= Ku + pixelSelector->currentPotential; i++)
    {
        for(int j = Kv-1; j <= Kv+1; j++)
        {
        	if(j < hG[0] && j >= 0 && i < wG[0] && i>=0)
            	currentFrame.at<uchar>(j, i) = 1;
        }
    }
}

void FullSystem::makeNewTraces(VisualState& visualState, FrameHessian* newFrame, float* gtDepth, const LidarProjectionResult& lidarProjection)
{
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel = lidarProjection.cloudPixels;
	float* selectionMapFromLidar = 0;

	cv::Mat mask = cv::Mat::zeros(hG[0], wG[0], CV_8UC1);

	pixelSelector->allowFast = true;

	int numPointsTotal = 0;
	int numPointLidar = 0;
	int numPointMonocular = 0;

	if(fabs(newFrame->shell->timestamp - lidarProjection.lidarTimestamp) < 0.01)
	{
		int left = wG[0], right = -1, up = hG[0], down = -1;
		for(size_t i = 0; i < vCloudPixel.size(); ++i)
		{
			const int u = static_cast<int>(vCloudPixel[i](0, 0));
			const int v = static_cast<int>(vCloudPixel[i](1, 0));
			if(u < left) left = u;
			if(u > right) right = u;
			if(v < up) up = v;
			if(v > down) down = v;
		}
		int lidarArea = (right >= left && down >= up) ? (right - left) * (down - up) : 0;
		int imageArea = wG[0] * hG[0];
	    selectionMapFromLidar = new float[vCloudPixel.size()];
	    numPointLidar = pixelSelector->makeMapsFromLidar(newFrame, selectionMapFromLidar, ((float)lidarArea/(float)imageArea) * setting_desiredImmatureDensity, 1, false, 1, vCloudPixel);

	    if(addFeaturePoint)
	    	numPointMonocular = pixelSelector->makeMaps(newFrame, visualState.selectionMap, setting_desiredImmatureDensity);
	    numPointsTotal = numPointLidar + numPointMonocular;
	}

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

	float maxScore = -1000.0;

	for(int i = 0; selectionMapFromLidar != 0 && i < vCloudPixel.size(); i++)
	{
		if(selectionMapFromLidar[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),newFrame, selectionMapFromLidar[i], &Hcalib);

		float score = shiTomasiScore(newFrame->dI, vCloudPixel[i](0, 0), vCloudPixel[i](1, 0));
		impt->score = score;
		if(score > maxScore) maxScore = score;

	    impt->idepth_fromSensor = 1.0 / vCloudPixel[i](2, 0);
		impt->idepth_max = impt->idepth_fromSensor * 1.0;
		impt->idepth_min = impt->idepth_fromSensor * 1.0;

		impt->isFromSensor = true;

	    if(!std::isfinite(impt->energyTH)) 
	    	delete impt;
		else{ 
			newFrame->immaturePoints.push_back(impt);
			setMask(mask, vCloudPixel[i](0, 0), vCloudPixel[i](1, 0));
		}
	}
	delete[] selectionMapFromLidar;

	float threshold = 0.01;
	for(ImmaturePoint* impt : newFrame->immaturePoints)
	{
		if(impt->score > threshold * maxScore)
			impt->type = ImmaturePoint::CORNER;
		else
			impt->type = ImmaturePoint::EDGELET;
	}

	if(addFeaturePoint)
	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(visualState.selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, visualState.selectionMap[i], &Hcalib);

		impt->isFromSensor = false;

		if(!std::isfinite(impt->energyTH) || mask.at<uchar>(y, x) == 1) 
			delete impt;
		else{ 
			newFrame->immaturePoints.push_back(impt);
			setMask(mask, x, y);
		}
	}

	mask.release();
}

void FullSystem::setPrecalcValues()
{
	setPrecalcValues(visualState);
}

void FullSystem::setPrecalcValues(VisualState& visualState)
{
	for(FrameHessian* fh : visualState.frameHessians)
	{
		fh->targetPrecalc.resize(visualState.frameHessians.size());
		for(unsigned int i=0;i<visualState.frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, visualState.frameHessians[i], &Hcalib);
	}

	visualState.ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	printLogLine(visualState);
}

void FullSystem::printLogLine(VisualState& visualState)
{
	if(visualState.frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                visualState.allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                visualState.ef->resInA,
                visualState.ef->resInL,
                visualState.ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                visualState.allKeyFramesHistory.back()->aff_g2l.a,
                visualState.allKeyFramesHistory.back()->aff_g2l.b,
                visualState.frameHessians.back()->shell->id - visualState.frameHessians.front()->shell->id,
                (int)visualState.frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << visualState.allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				visualState.ef->resInA << " "  <<
				visualState.ef->resInL << " "  <<
				visualState.ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				visualState.frameHessians.back()->aff_g2l().a << " "  <<
				visualState.frameHessians.back()->aff_g2l().b << " "  <<
				visualState.frameHessians.back()->shell->id - visualState.frameHessians.front()->shell->id << " "  <<
				(int)visualState.frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	printEigenValLine(visualState);
}

void FullSystem::printEigenValLine(VisualState& visualState)
{
	if(!setting_logStuff) return;
	if(visualState.ef->lastHS.rows() < 12) return;


	MatXX Hp = visualState.ef->lastHS.bottomRightCorner(visualState.ef->lastHS.cols()-CPARS,visualState.ef->lastHS.cols()-CPARS);
	MatXX Ha = visualState.ef->lastHS.bottomRightCorner(visualState.ef->lastHS.cols()-CPARS,visualState.ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = visualState.ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = visualState.ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << visualState.allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << visualState.allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << visualState.allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << visualState.allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = visualState.ef->lastHS.inverse().diagonal();
		(*variancesLog) << visualState.allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = visualState.ef->lastNullspaces_forLogging;
	(*nullspacesLog) << visualState.allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(visualState.ef->lastHS * nsp[i]) << " " << nsp[i].dot(visualState.ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	printFrameLifetimes(visualState);
}

void FullSystem::printFrameLifetimes(VisualState& visualState)
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : visualState.allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}

float FullSystem::shiTomasiScore(Eigen::Vector3f const *img, int u, int v)
{
	float k = 0.04;
  	float dXX = 0.0;
  	float dYY = 0.0;
  	float dXY = 0.0;
  	const int halfbox_size = 4;
  	const int box_size = 2*halfbox_size;
  	const int box_area = box_size*box_size;
  	const int x_min = u-halfbox_size;
  	const int x_max = u+halfbox_size;
  	const int y_min = v-halfbox_size;
  	const int y_max = v+halfbox_size;

  	if(x_min < 1 || x_max >= wG[0] - 1 || y_min < 1 || y_max >= hG[0] - 1)
    	return 0.0; // patch is too close to the boundary

  	const int stride = wG[0];
  	for( int y=y_min; y<y_max; ++y )
  	{
    	Eigen::Vector3f const* ptr_left   = img + stride*y + x_min - 1;
    	Eigen::Vector3f const* ptr_right  = img + stride*y + x_min + 1;
    	Eigen::Vector3f const* ptr_top    = img + stride*(y-1) + x_min;
    	Eigen::Vector3f const* ptr_bottom = img + stride*(y+1) + x_min;
    	for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
    	{
      		float dx = (*ptr_right)[0] - (*ptr_left)[0];
      		float dy = (*ptr_bottom)[0] - (*ptr_top)[0];
      		dXX += dx*dx;
      		dYY += dy*dy;
      		dXY += dx*dy;
    	}
  	}

  	// Find and return smaller eigenvalue:
  	dXX = dXX / (2.0 * box_area);
  	dYY = dYY / (2.0 * box_area);
  	dXY = dXY / (2.0 * box_area);

  	float lamda1 = 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
  	float lamda2 = 0.5 * (dXX + dYY + sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));

  	return (lamda1 * lamda2 - k * (lamda1 + lamda2) * (lamda1 + lamda2));
}

}
