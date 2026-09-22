#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <set>
#include <limits>
#include <execinfo.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>

#include <boost/bind.hpp>
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/FrameShell.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "system/ActiveCameraSelector.h"
#include "system/CameraFrameInput.h"
#include "system/CameraInput.h"
#include "system/LidarProjectionResult.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include "IOWrapper/Pangolin/PangolinViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "opencv2/highgui/highgui.hpp"

#include "ros/ros.h"
#include "ros/package.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>

typedef pcl::PointXYZI  PointType;

std::string vignette = "";
std::string gammaCalib = "";
std::string calib = "";
std::vector<std::string> calibPaths;
std::string resultPath = "";
std::string cameraScoreLogPath = "";
std::string pathSensorPrameter = "";
std::vector<std::string> pathSensorPrameters;
std::string imgTopic = "";
std::vector<std::string> imgTopics;
std::string lidarTopic = "";

double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;
bool preload=false;
bool useSampleOutput=false;
constexpr int kDefaultCameraId = 0;
double maxCameraLidarSyncError = 0.01;
double maxCoarseTrackingRMSE = 12.0;
double minCameraImageStddev = 3.0;
double minCameraImageGradientMean = 1.0;
double minCameraImageLaplacianVariance = 20.0;
double cameraSwitchScoreMargin = 1000.0;
int cameraSwitchConsecutiveFrames = 5;
double cameraSwitchMinInterval = 0.50;
double cameraResetSwitchCooldown = 1.00;
int minFallbackWarmupFrames = 4;
int minFallbackKeyframes = 1;
double maxFallbackReadyRMSE = 8.0;
bool useRobustTrackingPolicy = true;
int trackingStatsWindowSize = 20;
int minFallbackConsecutiveGoodFrames = 3;
int minFallbackTrackingObservations = 5;
double minFallbackTrackingSuccessRate = 0.70;
double maxFallbackTrackingRmseStddev = 4.0;
double maxFallbackPoseJumpRate = 0.20;
double maxTrackingPoseJumpMeters = 4.0;
double trackingConfidenceScoreWeight = 1200.0;
double trackingInstabilityPenaltyWeight = 800.0;
bool enableFallbackCameraRecovery = true;
int fallbackCameraRecoveryMinObservations = 20;
int fallbackCameraRecoveryConsecutiveRejects = 10;
double fallbackCameraRecoveryRejectedRate = 0.90;
double fallbackCameraRecoveryCooldown = 2.00;
double fallbackCameraQuarantineMinDuration = 1.00;
int fallbackCameraRecoveryStableFrames = 12;
double fallbackCameraRecoveryMinImageStddev = 40.0;
double fallbackCameraRecoveryMinGradientMean = 15.0;
int fallbackCameraRecoveryMinLidarPoints = 300;
double officialCameraMissingSwitchDelay = 0.20;
double maxFallbackCameraTimestampLag = 0.20;
int trackingCameraId = -1;
int officialCameraId = -1;
double lastOfficialFrameTimestamp = -1.0;
double lastCameraSwitchTimestamp = -1.0;
int pendingSwitchCameraId = -1;
int pendingSwitchCount = 0;
bool pendingPostSwitchRigAlignment = false;
int pendingPostSwitchCameraId = -1;
SE3 pendingPostSwitchRigPose;
std::map<int, double> lastCameraResetTimestampByCamera;
std::map<int, double> quarantinedFallbackCameraSinceByCamera;
std::map<int, int> fallbackCameraRecoveryGoodFramesByCamera;
int cameraEligibilityWarmupFrames = 30;
double eligibleCameraKeepRatio = 0.67;
int minEligibleCameras = 2;
int cameraEligibilitySuppressFrames = 5;
int cameraEligibilityRestoreFrames = 10;
double cameraEligibilityScoreAlpha = 0.20;
std::vector<double> cameraTimeOffsets;
std::map<int, unsigned long long> receivedImagesByCamera;
std::map<int, unsigned long long> receivedLidarProjectionsByCamera;
std::map<int, double> lastReceivedImageTimestampByCamera;
std::map<int, double> lastEnqueuedImageTimestampByCamera;
std::map<int, double> lastReceivedLidarTimestampByCamera;
std::map<int, double> lastEnqueuedLidarTimestampByCamera;
std::ofstream cameraScoreLogFile;

ImageFolderReader* reader = NULL;
std::vector<ImageFolderReader*> cameraReaders;
FullSystem* fullSystem = NULL;
bool firstFlag = true;
double firstFrameTime;
bool initialization = false;
double initialtimestamp = 0.0;
double timestamp = 0.0;
int currentId = 0;
constexpr int kResultSaveIntervalFrames = 50;
struct timeval tv_start;
clock_t started;
double sInitializerOffset=0;

static void segfaultHandler(int signal)
{
    void* trace[64];
    int traceSize = backtrace(trace, 64);
    fprintf(stderr, "\nFATAL: received signal %d. Backtrace:\n", signal);
    backtrace_symbols_fd(trace, traceSize, STDERR_FILENO);
    _exit(128 + signal);
}

pcl::PointCloud<PointType>::Ptr laserCloudIn;

pcl::PointCloud<PointType>::Ptr fullCloud;
pcl::PointCloud<PointType>::Ptr fullInfoCloud;

pcl::PointCloud<PointType>::Ptr groundCloud;
pcl::PointCloud<PointType>::Ptr segmentedCloud;

static bool acceptsCamera(int cameraId)
{
    return trackingCameraId < 0 || cameraId == trackingCameraId;
}

static std::string withSuffixBeforeExtensionLocal(const std::string& file, const std::string& suffix)
{
    const std::string::size_type slash = file.find_last_of("/\\");
    const std::string::size_type dot = file.find_last_of('.');
    if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return file.substr(0, dot) + suffix + file.substr(dot);
    return file + suffix;
}

static std::string withSuffixAndExtensionLocal(
    const std::string& file,
    const std::string& suffix,
    const std::string& extension)
{
    const std::string::size_type slash = file.find_last_of("/\\");
    const std::string::size_type dot = file.find_last_of('.');
    if(dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return file.substr(0, dot) + suffix + extension;
    return file + suffix + extension;
}

struct CameraEligibilityStats
{
    int observations;
    int validObservations;
    int topStreak;
    int outStreak;
    bool hasScore;
    bool eligible;
    double emaScore;

    CameraEligibilityStats()
        : observations(0),
          validObservations(0),
          topStreak(0),
          outStreak(0),
          hasScore(false),
          eligible(true),
          emaScore(0.0)
    {
    }
};

std::map<int, CameraEligibilityStats> cameraEligibilityStatsByCamera;
std::set<int> eligibleCameraSet;

static double clampDouble(double value, double lower, double upper)
{
    return std::max(lower, std::min(upper, value));
}

template<typename T>
static void pushBounded(std::deque<T>& values, const T& value, int maxSize)
{
    values.push_back(value);
    const int boundedSize = std::max(1, maxSize);
    while(static_cast<int>(values.size()) > boundedSize)
        values.pop_front();
}

static double meanDeque(const std::deque<double>& values)
{
    if(values.empty())
        return 0.0;
    double sum = 0.0;
    for(double value : values)
        sum += value;
    return sum / static_cast<double>(values.size());
}

static double stddevDeque(const std::deque<double>& values)
{
    if(values.size() < 2)
        return 0.0;
    const double mean = meanDeque(values);
    double accum = 0.0;
    for(double value : values)
    {
        const double delta = value - mean;
        accum += delta * delta;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
}

static double rateDeque(const std::deque<int>& values)
{
    if(values.empty())
        return 0.0;
    int count = 0;
    for(int value : values)
        count += value != 0 ? 1 : 0;
    return static_cast<double>(count) / static_cast<double>(values.size());
}

struct CameraTrackingStats
{
    int observations;
    int consecutiveGoodFrames;
    int consecutiveRejectedFrames;
    bool hasLastPosition;
    Eigen::Vector3d lastPosition;
    std::deque<double> rmseWindow;
    std::deque<int> successWindow;
    std::deque<int> rejectedWindow;
    std::deque<int> poseJumpWindow;

    CameraTrackingStats()
        : observations(0),
          consecutiveGoodFrames(0),
          consecutiveRejectedFrames(0),
          hasLastPosition(false),
          lastPosition(Eigen::Vector3d::Zero())
    {
    }

    double successRate() const { return rateDeque(successWindow); }
    double rejectedRate() const { return rateDeque(rejectedWindow); }
    double poseJumpRate() const { return rateDeque(poseJumpWindow); }
    double rmseMean() const { return meanDeque(rmseWindow); }
    double rmseStddev() const { return stddevDeque(rmseWindow); }

    double confidence() const
    {
        if(observations <= 0)
            return 0.0;

        const double rmseLimit = std::max(1e-6, maxFallbackReadyRMSE);
        const double rmseStddevLimit = std::max(1e-6, maxFallbackTrackingRmseStddev);
        const double goodStreakScale = std::max(1, minFallbackConsecutiveGoodFrames * 2);
        double confidenceValue = 0.0;
        confidenceValue += successRate() * 1.2;
        confidenceValue -= clampDouble(rmseMean() / rmseLimit, 0.0, 2.0) * 0.8;
        confidenceValue -= clampDouble(rmseStddev() / rmseStddevLimit, 0.0, 2.0) * 0.5;
        confidenceValue -= rejectedRate() * 1.0;
        confidenceValue -= poseJumpRate() * 1.0;
        confidenceValue +=
            clampDouble(static_cast<double>(consecutiveGoodFrames) / static_cast<double>(goodStreakScale), 0.0, 1.0) * 0.3;
        return clampDouble(confidenceValue, -2.0, 2.0);
    }
};

std::map<int, CameraTrackingStats> cameraTrackingStatsByCamera;

static std::vector<int> getConfiguredCameraIds()
{
    std::vector<int> cameraIds;
    if(!imgTopics.empty())
    {
        for(size_t i = 0; i < imgTopics.size(); ++i)
            cameraIds.push_back(static_cast<int>(i));
    }
    else if(!cameraReaders.empty())
    {
        for(size_t i = 0; i < cameraReaders.size(); ++i)
            cameraIds.push_back(static_cast<int>(i));
    }

    if(cameraIds.empty())
    {
        for(const std::pair<const int, CameraEligibilityStats>& entry : cameraEligibilityStatsByCamera)
            cameraIds.push_back(entry.first);
    }

    return cameraIds;
}

static bool isCameraEligible(int cameraId)
{
    if(trackingCameraId >= 0)
        return cameraId == trackingCameraId;
    if(eligibleCameraSet.empty())
        return true;
    return eligibleCameraSet.count(cameraId) > 0;
}

static void printCameraEligibilityUpdate(
    const std::vector<int>& cameraIds,
    const std::set<int>& previousEligibleSet)
{
    std::set<int> currentEligibleSet = eligibleCameraSet;
    if(currentEligibleSet == previousEligibleSet)
        return;

    printf("camera eligibility update: keep [");
    bool first = true;
    for(int cameraId : cameraIds)
    {
        if(currentEligibleSet.count(cameraId) == 0)
            continue;
        if(!first) printf(",");
        printf("%d", cameraId);
        first = false;
    }
    printf("] suppress [");
    first = true;
    for(int cameraId : cameraIds)
    {
        if(currentEligibleSet.count(cameraId) > 0)
            continue;
        if(!first) printf(",");
        const CameraEligibilityStats& stats = cameraEligibilityStatsByCamera[cameraId];
        const double validRate = stats.observations > 0
            ? static_cast<double>(stats.validObservations) / static_cast<double>(stats.observations)
            : 0.0;
        printf("%d(avg=%.1f valid=%.2f)", cameraId, stats.hasScore ? stats.emaScore : -std::numeric_limits<double>::infinity(), validRate);
        first = false;
    }
    printf("]\n");
}

static void updateCameraEligibility(
    const std::vector<sdv_loam::CameraFrameInput>& candidates,
    const std::map<int, sdv_loam::CameraSelectionScore>& validScoresByCamera)
{
    std::set<int> previousEligibleSet = eligibleCameraSet;

    for(const sdv_loam::CameraFrameInput& candidate : candidates)
    {
        CameraEligibilityStats& stats = cameraEligibilityStatsByCamera[candidate.cameraId];
        stats.observations++;

        const std::map<int, sdv_loam::CameraSelectionScore>::const_iterator scoreIt =
            validScoresByCamera.find(candidate.cameraId);
        const bool validScore = scoreIt != validScoresByCamera.end();
        const double observedScore = validScore ? scoreIt->second.score : -10000.0;
        if(validScore)
            stats.validObservations++;

        if(!stats.hasScore)
        {
            stats.emaScore = observedScore;
            stats.hasScore = true;
        }
        else
        {
            const double alpha = std::min(1.0, std::max(0.0, cameraEligibilityScoreAlpha));
            stats.emaScore = (1.0 - alpha) * stats.emaScore + alpha * observedScore;
        }
    }

    std::vector<int> cameraIds = getConfiguredCameraIds();
    if(cameraIds.empty())
        return;

    for(int cameraId : cameraIds)
        cameraEligibilityStatsByCamera[cameraId];

    const int totalCameras = static_cast<int>(cameraIds.size());
    const int clampedMinEligible = std::min(totalCameras, std::max(1, minEligibleCameras));
    if(trackingCameraId >= 0 ||
       currentId < cameraEligibilityWarmupFrames ||
       totalCameras <= clampedMinEligible)
    {
        eligibleCameraSet.clear();
        for(int cameraId : cameraIds)
            eligibleCameraSet.insert(cameraId);
        printCameraEligibilityUpdate(cameraIds, previousEligibleSet);
        return;
    }

    const int keepCount = std::min(
        totalCameras,
        std::max(clampedMinEligible, static_cast<int>(std::ceil(totalCameras * eligibleCameraKeepRatio))));

    std::vector<std::pair<double, int> > rankedCameras;
    rankedCameras.reserve(cameraIds.size());
    for(int cameraId : cameraIds)
    {
        const CameraEligibilityStats& stats = cameraEligibilityStatsByCamera[cameraId];
        const double score = stats.hasScore ? stats.emaScore : -std::numeric_limits<double>::infinity();
        rankedCameras.push_back(std::make_pair(score, cameraId));
    }

    std::sort(rankedCameras.begin(), rankedCameras.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            if(a.first == b.first) return a.second < b.second;
            return a.first > b.first;
        });

    std::set<int> topCameraSet;
    for(int i = 0; i < keepCount && i < static_cast<int>(rankedCameras.size()); ++i)
        topCameraSet.insert(rankedCameras[i].second);

    for(int cameraId : cameraIds)
    {
        CameraEligibilityStats& stats = cameraEligibilityStatsByCamera[cameraId];
        if(topCameraSet.count(cameraId) > 0)
        {
            stats.topStreak++;
            stats.outStreak = 0;
            if(!stats.eligible && stats.topStreak >= std::max(1, cameraEligibilityRestoreFrames))
                stats.eligible = true;
        }
        else
        {
            stats.outStreak++;
            stats.topStreak = 0;
            if(stats.eligible && stats.outStreak >= std::max(1, cameraEligibilitySuppressFrames))
                stats.eligible = false;
        }
    }

    eligibleCameraSet.clear();
    for(int cameraId : cameraIds)
    {
        if(cameraEligibilityStatsByCamera[cameraId].eligible)
            eligibleCameraSet.insert(cameraId);
    }

    if(static_cast<int>(eligibleCameraSet.size()) < keepCount)
    {
        for(int cameraId : topCameraSet)
        {
            cameraEligibilityStatsByCamera[cameraId].eligible = true;
            eligibleCameraSet.insert(cameraId);
        }
    }

    printCameraEligibilityUpdate(cameraIds, previousEligibleSet);
}

static void applyTrackingStatsToScore(int cameraId, sdv_loam::CameraSelectionScore& score)
{
    if(!useRobustTrackingPolicy)
        return;

    std::map<int, CameraTrackingStats>::const_iterator statsIt =
        cameraTrackingStatsByCamera.find(cameraId);
    if(statsIt == cameraTrackingStatsByCamera.end() || statsIt->second.observations <= 0)
        return;

    const CameraTrackingStats& stats = statsIt->second;
    score.score += trackingConfidenceScoreWeight * stats.confidence();
    score.score -= trackingInstabilityPenaltyWeight * stats.rejectedRate();
    score.score -= trackingInstabilityPenaltyWeight * stats.poseJumpRate();

    if(stats.observations >= minFallbackTrackingObservations &&
       stats.consecutiveGoodFrames < minFallbackConsecutiveGoodFrames)
    {
        score.score -= 0.5 * trackingInstabilityPenaltyWeight;
    }
}

static bool isCameraTrackingStable(int cameraId)
{
    if(!useRobustTrackingPolicy)
        return true;

    std::map<int, CameraTrackingStats>::const_iterator statsIt =
        cameraTrackingStatsByCamera.find(cameraId);
    if(statsIt == cameraTrackingStatsByCamera.end())
        return false;

    const CameraTrackingStats& stats = statsIt->second;
    if(stats.observations < std::max(1, minFallbackTrackingObservations))
        return false;
    if(stats.consecutiveGoodFrames < std::max(1, minFallbackConsecutiveGoodFrames))
        return false;
    if(stats.successRate() < minFallbackTrackingSuccessRate)
        return false;
    if(stats.rmseStddev() > maxFallbackTrackingRmseStddev)
        return false;
    if(stats.poseJumpRate() > maxFallbackPoseJumpRate)
        return false;

    return true;
}

static void updateCameraTrackingStats(int cameraId, size_t frameHistorySizeBefore)
{
    if(!useRobustTrackingPolicy || fullSystem == NULL)
        return;

    sdv_loam::VisualState& state = fullSystem->getVisualState(cameraId);
    CameraTrackingStats& stats = cameraTrackingStatsByCamera[cameraId];
    const size_t frameHistorySizeAfter = state.allFrameHistory.size();
    const bool acceptedFrame = frameHistorySizeAfter > frameHistorySizeBefore;
    const bool finiteRmse = std::isfinite(state.lastCoarseRMSE[0]);
    const double observedRmse = finiteRmse
        ? static_cast<double>(state.lastCoarseRMSE[0])
        : maxCoarseTrackingRMSE * 2.0;

    bool poseJump = false;
    if(acceptedFrame && !state.allFrameHistory.empty())
    {
        const sdv_loam::FrameShell* shell = state.allFrameHistory.back();
        if(shell != 0 && shell->poseValid)
        {
            const SE3 rigPose = fullSystem->getRigPoseForFrameShell(shell);
            const Eigen::Vector3d position = rigPose.translation();
            if(stats.hasLastPosition)
            {
                const double step = (position - stats.lastPosition).norm();
                poseJump = maxTrackingPoseJumpMeters > 0.0 && step > maxTrackingPoseJumpMeters;
            }
            stats.lastPosition = position;
            stats.hasLastPosition = true;
        }
    }

    const bool goodFrame =
        acceptedFrame &&
        state.initialized &&
        !state.isLost &&
        !state.initFailed &&
        finiteRmse &&
        observedRmse <= maxFallbackReadyRMSE &&
        !poseJump;

    stats.observations++;
    stats.consecutiveGoodFrames = goodFrame ? stats.consecutiveGoodFrames + 1 : 0;
    stats.consecutiveRejectedFrames = acceptedFrame ? 0 : stats.consecutiveRejectedFrames + 1;
    pushBounded(stats.rmseWindow, observedRmse, trackingStatsWindowSize);
    pushBounded(stats.successWindow, goodFrame ? 1 : 0, trackingStatsWindowSize);
    pushBounded(stats.rejectedWindow, acceptedFrame ? 0 : 1, trackingStatsWindowSize);
    pushBounded(stats.poseJumpWindow, poseJump ? 1 : 0, trackingStatsWindowSize);
}

static bool cameraPassesRecoveryImageGate(const sdv_loam::CameraSelectionScore& score)
{
    return score.valid &&
           score.imageStddev >= fallbackCameraRecoveryMinImageStddev &&
           score.imageGradientMean >= fallbackCameraRecoveryMinGradientMean &&
           score.projectedLidarPoints >= fallbackCameraRecoveryMinLidarPoints;
}

static bool isFallbackCameraQuarantined(int cameraId, double timestamp)
{
    const std::map<int, double>::const_iterator it = quarantinedFallbackCameraSinceByCamera.find(cameraId);
    if(it == quarantinedFallbackCameraSinceByCamera.end())
        return false;

    if(fallbackCameraQuarantineMinDuration < 0.0 || timestamp < 0.0)
        return true;

    return timestamp - it->second < fallbackCameraQuarantineMinDuration ||
           fallbackCameraRecoveryGoodFramesByCamera[cameraId] < std::max(1, fallbackCameraRecoveryStableFrames);
}

static void updateFallbackCameraRecoveryState(
    int cameraId,
    const sdv_loam::CameraSelectionScore& score,
    double timestamp)
{
    if(!enableFallbackCameraRecovery || fullSystem == NULL)
        return;
    if(officialCameraId < 0 || cameraId == officialCameraId)
        return;
    if(!score.valid)
        return;

    const std::map<int, CameraTrackingStats>::const_iterator statsIt =
        cameraTrackingStatsByCamera.find(cameraId);
    if(statsIt == cameraTrackingStatsByCamera.end())
        return;

    const CameraTrackingStats& stats = statsIt->second;
    if(stats.observations < std::max(1, fallbackCameraRecoveryMinObservations))
        return;

    const bool persistentlyRejected =
        stats.rejectedRate() >= fallbackCameraRecoveryRejectedRate ||
        stats.consecutiveRejectedFrames >= std::max(1, fallbackCameraRecoveryConsecutiveRejects);
    const bool inQuarantine = quarantinedFallbackCameraSinceByCamera.count(cameraId) > 0;

    if(persistentlyRejected)
    {
        if(!inQuarantine)
        {
            quarantinedFallbackCameraSinceByCamera[cameraId] = timestamp;
            fallbackCameraRecoveryGoodFramesByCamera[cameraId] = 0;
            pendingSwitchCameraId = -1;
            pendingSwitchCount = 0;
            printf("quarantined fallback camera %d (rejected_rate=%.2f consecutive_rejected=%d image_stddev=%.2f gradient=%.2f lidar_points=%d).\n",
                   cameraId,
                   stats.rejectedRate(),
                   stats.consecutiveRejectedFrames,
                   score.imageStddev,
                   score.imageGradientMean,
                   score.projectedLidarPoints);
        }
        return;
    }

    if(!inQuarantine)
        return;

    const bool imageRecovered = cameraPassesRecoveryImageGate(score);
    const bool trackingRecovered = isCameraTrackingStable(cameraId);
    if(imageRecovered && trackingRecovered)
        fallbackCameraRecoveryGoodFramesByCamera[cameraId]++;
    else
        fallbackCameraRecoveryGoodFramesByCamera[cameraId] = 0;

    const double quarantineDuration =
        timestamp >= 0.0 ? timestamp - quarantinedFallbackCameraSinceByCamera[cameraId] : 0.0;
    const bool durationReady =
        fallbackCameraQuarantineMinDuration < 0.0 || quarantineDuration >= fallbackCameraQuarantineMinDuration;
    const bool stableReady =
        fallbackCameraRecoveryGoodFramesByCamera[cameraId] >= std::max(1, fallbackCameraRecoveryStableFrames);

    if(durationReady && stableReady)
    {
        printf("released fallback camera %d from quarantine (duration=%.2f s stable_frames=%d score=%.1f).\n",
               cameraId,
               quarantineDuration,
               fallbackCameraRecoveryGoodFramesByCamera[cameraId],
               score.score);
        quarantinedFallbackCameraSinceByCamera.erase(cameraId);
        fallbackCameraRecoveryGoodFramesByCamera.erase(cameraId);
    }
}

static std::string getCameraScoreLogPath()
{
    if(!cameraScoreLogPath.empty())
        return cameraScoreLogPath;
    if(resultPath.empty())
        return "";
    return withSuffixAndExtensionLocal(resultPath, "_camera_scores", ".csv");
}

static bool ensureCameraScoreLogOpen()
{
    if(cameraScoreLogFile.is_open())
        return true;

    const std::string path = getCameraScoreLogPath();
    if(path.empty())
        return false;

    cameraScoreLogFile.open(path.c_str());
    if(!cameraScoreLogFile.is_open())
    {
        printf("warning: could not open camera score log at %s\n", path.c_str());
        return false;
    }

    cameraScoreLogFile
        << "frame_id,"
        << "timestamp,"
        << "image_timestamp,"
        << "lidar_timestamp,"
        << "camera,"
        << "selected_camera,"
        << "role,"
        << "score_valid,"
        << "score,"
        << "lidar_points,"
        << "sync_error,"
        << "image_mean,"
        << "image_stddev,"
        << "image_gradient_mean,"
        << "image_laplacian_variance,"
        << "tracking_rmse,"
        << "initialized,"
        << "is_lost,"
        << "init_failed,"
        << "active_keyframes,"
        << "total_frames,"
        << "eligible,"
        << "invalid_quality,"
        << "tracking_observations,"
        << "tracking_consecutive_good,"
        << "tracking_consecutive_rejected,"
        << "tracking_success_rate,"
        << "tracking_rejected_rate,"
        << "tracking_pose_jump_rate,"
        << "tracking_rmse_mean,"
        << "tracking_rmse_stddev,"
        << "tracking_stable"
        << "\n";
    printf("camera score log: %s\n", path.c_str());
    return true;
}

static void writeCameraScoreLog(
    const std::vector<sdv_loam::CameraFrameInput>& candidates,
    const std::map<int, sdv_loam::CameraSelectionScore>& scoresByCamera,
    const std::set<int>& invalidQualityCameraSet,
    int selectedCameraId,
    double latestCandidateTimestamp)
{
    if(!ensureCameraScoreLogOpen())
        return;

    cameraScoreLogFile << std::fixed << std::setprecision(9);

    for(const sdv_loam::CameraFrameInput& candidate : candidates)
    {
        const std::map<int, sdv_loam::CameraSelectionScore>::const_iterator scoreIt =
            scoresByCamera.find(candidate.cameraId);
        const bool hasScore = scoreIt != scoresByCamera.end();
        const sdv_loam::CameraSelectionScore score =
            hasScore ? scoreIt->second : sdv_loam::CameraSelectionScore();
        const std::map<int, CameraTrackingStats>::const_iterator statsIt =
            cameraTrackingStatsByCamera.find(candidate.cameraId);
        const bool hasStats = statsIt != cameraTrackingStatsByCamera.end();
        const CameraTrackingStats* stats = hasStats ? &statsIt->second : 0;
        const bool selected = candidate.cameraId == selectedCameraId;
        const bool eligible = isCameraEligible(candidate.cameraId);
        const bool invalidQuality = invalidQualityCameraSet.count(candidate.cameraId) > 0;

        std::string role = "candidate";
        if(selected)
        {
            role = "active";
        }
        else if(score.valid && eligible && !invalidQuality)
        {
            const bool timeAligned =
                selectedCameraId < 0 ||
                maxFallbackCameraTimestampLag < 0.0 ||
                latestCandidateTimestamp < 0.0 ||
                latestCandidateTimestamp - candidate.imageTimestamp <= maxFallbackCameraTimestampLag;
            role = timeAligned ? "warmup" : "stale";
        }
        else if(invalidQuality)
        {
            role = "invalid_quality";
        }
        else if(!eligible)
        {
            role = "suppressed";
        }
        else if(!score.valid)
        {
            role = "invalid_score";
        }

        cameraScoreLogFile
            << currentId << ","
            << candidate.imageTimestamp << ","
            << candidate.imageTimestamp << ","
            << candidate.lidarProjection.lidarTimestamp << ","
            << candidate.cameraId << ","
            << selectedCameraId << ","
            << role << ","
            << (score.valid ? 1 : 0) << ","
            << score.score << ","
            << candidate.lidarProjection.cloudPixels.size() << ","
            << std::fabs(candidate.imageTimestamp - candidate.lidarProjection.lidarTimestamp) << ","
            << score.imageMean << ","
            << score.imageStddev << ","
            << score.imageGradientMean << ","
            << score.imageLaplacianVariance << ","
            << score.trackingRmse << ","
            << (score.initialized ? 1 : 0) << ","
            << (score.isLost ? 1 : 0) << ","
            << (score.initFailed ? 1 : 0) << ","
            << score.activeKeyframes << ","
            << score.totalFrames << ","
            << (eligible ? 1 : 0) << ","
            << (invalidQuality ? 1 : 0) << ","
            << (stats ? stats->observations : 0) << ","
            << (stats ? stats->consecutiveGoodFrames : 0) << ","
            << (stats ? stats->consecutiveRejectedFrames : 0) << ","
            << (stats ? stats->successRate() : 0.0) << ","
            << (stats ? stats->rejectedRate() : 0.0) << ","
            << (stats ? stats->poseJumpRate() : 0.0) << ","
            << (stats ? stats->rmseMean() : 0.0) << ","
            << (stats ? stats->rmseStddev() : 0.0) << ","
            << (isCameraTrackingStable(candidate.cameraId) ? 1 : 0)
            << "\n";
    }
    cameraScoreLogFile.flush();
}

static double getCameraTimeOffset(int cameraId)
{
    if(cameraId >= 0 && cameraId < static_cast<int>(cameraTimeOffsets.size()))
        return cameraTimeOffsets[cameraId];

    return 0.0;
}

static void applyRuntimeFullSystemSettings()
{
    if(fullSystem != NULL)
        fullSystem->maxCoarseTrackingRMSE = static_cast<float>(maxCoarseTrackingRMSE);
}

static ImageFolderReader* getReaderForCamera(int cameraId)
{
    if(cameraId >= 0 && cameraId < static_cast<int>(cameraReaders.size()) && cameraReaders[cameraId] != NULL)
        return cameraReaders[cameraId];

    return reader;
}

static void applyVisualCalibrationForCamera(int cameraId)
{
    ImageFolderReader* cameraReader = getReaderForCamera(cameraId);
    if(fullSystem == NULL || cameraReader == NULL)
        return;

    Eigen::Matrix3f K;
    int w = 0;
    int h = 0;
    cameraReader->getCalibMono(K, w, h);
    if(w != wG[0] || h != hG[0])
    {
        printf("WARNING: camera %d calib resolution %d x %d differs from global %d x %d.\n",
               cameraId, w, h, wG[0], hG[0]);
    }

    fullSystem->setVisualCalibration(cameraId, K);
}

static void createCameraReaders()
{
    if(calibPaths.empty() && !calib.empty())
        calibPaths.push_back(calib);

    if(calibPaths.empty())
        return;

    cameraReaders.resize(calibPaths.size(), NULL);
    for(size_t cameraId = 0; cameraId < calibPaths.size(); ++cameraId)
    {
        cameraReaders[cameraId] = new ImageFolderReader(calibPaths[cameraId], gammaCalib, vignette);
        printf("created camera %zu visual reader from %s\n", cameraId, calibPaths[cameraId].c_str());
    }

    reader = cameraReaders[0];
}

static void applyAllCameraIntrinsics()
{
    if(fullSystem == NULL)
        return;

    for(size_t cameraId = 0; cameraId < cameraReaders.size(); ++cameraId)
    {
        if(cameraReaders[cameraId] == NULL)
            continue;

        Eigen::Matrix3f K;
        int w = 0;
        int h = 0;
        cameraReaders[cameraId]->getCalibMono(K, w, h);
        fullSystem->setCameraIntrinsics(static_cast<int>(cameraId), K(0, 0), K(1, 1), K(0, 2), K(1, 2));
    }
}

static void printNoSyncDiagnostics()
{
    static int noSyncCounter = 0;
    noSyncCounter++;
    if(noSyncCounter % 60 != 0 || fullSystem == NULL)
        return;

    printf("waiting for synchronized camera/lidar candidate (gate %.6f s)\n", maxCameraLidarSyncError);

    const std::vector<int> cameraIds = fullSystem->getConfiguredCameraIds();
    for(size_t i = 0; i < cameraIds.size(); ++i)
    {
        const int cameraId = cameraIds[i];
        if(!acceptsCamera(cameraId))
            continue;

        const size_t imgQueueSize = fullSystem->qTimeImgByCamera[cameraId].size();
        const size_t lidarQueueSize = fullSystem->qLidarProjectionResultsByCamera[cameraId].size();
        printf("  camera %d received: img=%llu lidar=%llu queues: img=%zu lidar=%zu",
               cameraId,
               receivedImagesByCamera[cameraId],
               receivedLidarProjectionsByCamera[cameraId],
               imgQueueSize,
               lidarQueueSize);

        printf(" last_img_rx=%.6f last_img_q=%.6f last_lidar_rx=%.6f last_lidar_q=%.6f drops: img_old=%llu lidar_old=%llu",
               lastReceivedImageTimestampByCamera[cameraId],
               lastEnqueuedImageTimestampByCamera[cameraId],
               lastReceivedLidarTimestampByCamera[cameraId],
               lastEnqueuedLidarTimestampByCamera[cameraId],
               fullSystem->droppedImagesTooOldByCamera[cameraId],
               fullSystem->droppedLidarTooOldByCamera[cameraId]);

        if(imgQueueSize > 0 && lidarQueueSize > 0)
        {
            const double imageTimestamp = fullSystem->qTimeImgByCamera[cameraId].front();
            const double lidarTimestamp = fullSystem->qLidarProjectionResultsByCamera[cameraId].front().lidarTimestamp;
            printf(" front_img=%.6f front_lidar=%.6f diff=%.6f offset=%+.6f",
                   imageTimestamp,
                   lidarTimestamp,
                   fabs(imageTimestamp - lidarTimestamp),
                   getCameraTimeOffset(cameraId));
        }

        printf("\n");
    }
}

static bool hasSynchronizedCameraCandidates()
{
    const std::vector<int> cameraIds = fullSystem->getSynchronizedCameraCandidateIds(maxCameraLidarSyncError);
    for(size_t i = 0; i < cameraIds.size(); ++i)
        if(acceptsCamera(cameraIds[i]))
            return true;

    return false;
}

static std::vector<sdv_loam::CameraFrameInput> buildCurrentCameraCandidates()
{
    std::vector<sdv_loam::CameraFrameInput> candidates;
    const std::vector<int> cameraIds = fullSystem->getSynchronizedCameraCandidateIds(maxCameraLidarSyncError);
    candidates.reserve(cameraIds.size());

    for(size_t i = 0; i < cameraIds.size(); ++i)
    {
        if(acceptsCamera(cameraIds[i]))
            candidates.push_back(fullSystem->buildQueuedCameraCandidate(cameraIds[i]));
    }

    return candidates;
}

static int dropStaleCameraCandidates(int cameraId, double minTimestamp)
{
    int dropped = 0;
    while(fullSystem->synchronizeQueuedCameraCandidate(cameraId, maxCameraLidarSyncError))
    {
        const sdv_loam::CameraFrameInput candidate = fullSystem->buildQueuedCameraCandidate(cameraId);
        if(candidate.imageTimestamp >= minTimestamp)
            break;

        fullSystem->popQueuedCameraCandidate(cameraId);
        dropped++;
    }

    return dropped;
}

static void dropStaleCameraCandidatesExcept(int activeCameraId, double referenceTimestamp)
{
    if(referenceTimestamp < 0.0 || maxFallbackCameraTimestampLag < 0.0)
        return;

    const double minTimestamp = referenceTimestamp - maxFallbackCameraTimestampLag;
    const std::vector<int> cameraIds = fullSystem->getConfiguredCameraIds();
    for(size_t i = 0; i < cameraIds.size(); ++i)
    {
        const int cameraId = cameraIds[i];
        if(cameraId == activeCameraId || !acceptsCamera(cameraId))
            continue;

        const int dropped = dropStaleCameraCandidates(cameraId, minTimestamp);
        if(dropped > 0)
        {
            printf("dropped %d stale camera candidates camera=%d older_than=%.6f reference=%.6f max_lag=%.3f\n",
                   dropped,
                   cameraId,
                   minTimestamp,
                   referenceTimestamp,
                   maxFallbackCameraTimestampLag);
        }
    }
}

static void configureImageCameras()
{
    if(imgTopics.empty())
    {
        fullSystem->setActiveCameraId(kDefaultCameraId);
        return;
    }

    for(size_t cameraId = 0; cameraId < imgTopics.size(); ++cameraId)
        fullSystem->setActiveCameraId(static_cast<int>(cameraId));

    fullSystem->setActiveCameraId(trackingCameraId >= 0 ? trackingCameraId : kDefaultCameraId);
}

static void loadCameraSensorParameters()
{
    if(!pathSensorPrameters.empty())
    {
        for(size_t cameraId = 0; cameraId < pathSensorPrameters.size(); ++cameraId)
            fullSystem->loadSensorPrameters(static_cast<int>(cameraId), pathSensorPrameters[cameraId]);

        fullSystem->setActiveCameraId(trackingCameraId >= 0 ? trackingCameraId : kDefaultCameraId);
        return;
    }

    if(!pathSensorPrameter.empty())
        fullSystem->loadSensorPrameters(kDefaultCameraId, pathSensorPrameter);
}

static std::string resolveRosFindSubstitutions(const std::string& value)
{
    std::string resolved = value;
    const std::string marker = "$(find ";
    size_t pos = resolved.find(marker);

    while(pos != std::string::npos)
    {
        const size_t packageStart = pos + marker.size();
        const size_t packageEnd = resolved.find(")", packageStart);
        if(packageEnd == std::string::npos)
            break;

        const std::string packageName = resolved.substr(packageStart, packageEnd - packageStart);
        const std::string packagePath = ros::package::getPath(packageName);
        if(packagePath.empty())
            break;

        resolved.replace(pos, packageEnd - pos + 1, packagePath);
        pos = resolved.find(marker, pos + packagePath.size());
    }

    return resolved;
}
pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
pcl::PointCloud<PointType>::Ptr outlierCloud;

PointType nanPoint;

cv::Mat rangeMat;
cv::Mat labelMat;
cv::Mat groundMat;
int labelCount;

float startOrientation;
float endOrientation;

std::vector<std::pair<int8_t, int8_t> > neighborIterator;

uint16_t *allPushedIndX;
uint16_t *allPushedIndY;

uint16_t *queueIndX;
uint16_t *queueIndY;

//Vel 64 Parameters
extern const int N_SCAN = 64;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 0.427;
extern const float ang_bottom = 24.9;
extern const int groundScanInd = 50;

extern const bool loopClosureEnableFlag = false;
extern const double mappingProcessInterval = 0.3;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;

extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 60.0/180.0*M_PI;
extern const int segmentValidPointNum = 5;
extern const int segmentValidLineNum = 3;
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;

void allocateMemory(){
    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloud.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud.reset(new pcl::PointCloud<PointType>());

    groundCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
    outlierCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

    allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
}

void resetParameters(){
    laserCloudIn->clear();
    groundCloud->clear();
    segmentedCloud->clear();
    segmentedCloudPure->clear();
    outlierCloud->clear();

    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
}

int mode=0;

bool firstRosSpin=false;

using namespace sdv_loam;

void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}

void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}




void parseArgument(ros::NodeHandle &n)
{
	int option;
	float foption;
	char buf[1000];

	int sampleoutput_msg;
    if(n.getParam("sampleoutput", sampleoutput_msg))
    {
        if(sampleoutput_msg==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
    }

    int quiet_msg;
    if(n.getParam("quiet", quiet_msg))
    {
        if(quiet_msg==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
    }

    int preset_msg;
    if(n.getParam("preset", preset_msg))
	{
		settingsDefault(preset_msg);
	}

    int rec_msg;
	if(n.getParam("rec", rec_msg))
	{
		if(rec_msg==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
	}

    int noros_msg;
	if(n.getParam("noros", noros_msg))
	{
		if(noros_msg==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
	}

    int nolog_msg;
	if(n.getParam("nolog", nolog_msg))
	{
		if(nolog_msg==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
	}

	int reverse_msg;
	if(n.getParam("reverse", reverse_msg))
	{
		if(reverse_msg==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
	}

	int nogui_msg;
	if(n.getParam("nogui", nogui_msg))
	{
		if(nogui_msg==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
	}

	int nomt_msg;
	if(n.getParam("nomt", nomt_msg))
	{
		if(nomt_msg==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
	}

	int prefetch_msg;
	if(n.getParam("prefetch", prefetch_msg))
	{
		if(prefetch_msg==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
	}

	int start_msg;
	if(n.getParam("start", start_msg))
	{
		start = start_msg;
		printf("START AT %d!\n",start);
	}

	int end_msg;
	if(n.getParam("end", end_msg))
	{
		end = end_msg;
		printf("END AT %d!\n",start);
	}

	int trackingCameraId_msg;
	if(n.getParam("trackingCameraId", trackingCameraId_msg))
	{
		trackingCameraId = trackingCameraId_msg;
		printf("tracking visual core with camera %d!\n", trackingCameraId);
	}

	double maxCameraLidarSyncError_msg;
	if(n.getParam("maxCameraLidarSyncError", maxCameraLidarSyncError_msg))
	{
		maxCameraLidarSyncError = maxCameraLidarSyncError_msg;
		printf("max camera-lidar sync error %f seconds!\n", maxCameraLidarSyncError);
	}

	double maxCoarseTrackingRMSE_msg;
	if(n.getParam("maxCoarseTrackingRMSE", maxCoarseTrackingRMSE_msg))
	{
		maxCoarseTrackingRMSE = maxCoarseTrackingRMSE_msg;
		printf("max coarse tracking RMSE %f!\n", maxCoarseTrackingRMSE);
	}

	double minCameraImageStddev_msg;
	if(n.getParam("minCameraImageStddev", minCameraImageStddev_msg))
	{
		minCameraImageStddev = minCameraImageStddev_msg;
		printf("min camera image stddev %f!\n", minCameraImageStddev);
	}

	double minCameraImageGradientMean_msg;
	if(n.getParam("minCameraImageGradientMean", minCameraImageGradientMean_msg))
	{
		minCameraImageGradientMean = minCameraImageGradientMean_msg;
		printf("min camera image gradient mean %f!\n", minCameraImageGradientMean);
	}

	double minCameraImageLaplacianVariance_msg;
	if(n.getParam("minCameraImageLaplacianVariance", minCameraImageLaplacianVariance_msg))
	{
		minCameraImageLaplacianVariance = minCameraImageLaplacianVariance_msg;
		printf("min camera image laplacian variance %f!\n", minCameraImageLaplacianVariance);
	}

	double cameraSwitchScoreMargin_msg;
	if(n.getParam("cameraSwitchScoreMargin", cameraSwitchScoreMargin_msg))
	{
		cameraSwitchScoreMargin = cameraSwitchScoreMargin_msg;
		printf("camera switch score margin %f!\n", cameraSwitchScoreMargin);
	}

	int cameraSwitchConsecutiveFrames_msg;
	if(n.getParam("cameraSwitchConsecutiveFrames", cameraSwitchConsecutiveFrames_msg))
	{
		cameraSwitchConsecutiveFrames = cameraSwitchConsecutiveFrames_msg;
		printf("camera switch consecutive frames %d!\n", cameraSwitchConsecutiveFrames);
	}

	double cameraSwitchMinInterval_msg;
	if(n.getParam("cameraSwitchMinInterval", cameraSwitchMinInterval_msg))
	{
		cameraSwitchMinInterval = cameraSwitchMinInterval_msg;
		printf("camera switch min interval %f seconds!\n", cameraSwitchMinInterval);
	}

	double cameraResetSwitchCooldown_msg;
	if(n.getParam("cameraResetSwitchCooldown", cameraResetSwitchCooldown_msg))
	{
		cameraResetSwitchCooldown = cameraResetSwitchCooldown_msg;
		printf("camera reset switch cooldown %f seconds!\n", cameraResetSwitchCooldown);
	}

	int cameraEligibilityWarmupFrames_msg;
	if(n.getParam("cameraEligibilityWarmupFrames", cameraEligibilityWarmupFrames_msg))
	{
		cameraEligibilityWarmupFrames = cameraEligibilityWarmupFrames_msg;
		printf("camera eligibility warmup frames %d!\n", cameraEligibilityWarmupFrames);
	}

	double eligibleCameraKeepRatio_msg;
	if(n.getParam("eligibleCameraKeepRatio", eligibleCameraKeepRatio_msg))
	{
		eligibleCameraKeepRatio = eligibleCameraKeepRatio_msg;
		printf("eligible camera keep ratio %f!\n", eligibleCameraKeepRatio);
	}

	int minEligibleCameras_msg;
	if(n.getParam("minEligibleCameras", minEligibleCameras_msg))
	{
		minEligibleCameras = minEligibleCameras_msg;
		printf("min eligible cameras %d!\n", minEligibleCameras);
	}

	int cameraEligibilitySuppressFrames_msg;
	if(n.getParam("cameraEligibilitySuppressFrames", cameraEligibilitySuppressFrames_msg))
	{
		cameraEligibilitySuppressFrames = cameraEligibilitySuppressFrames_msg;
		printf("camera eligibility suppress frames %d!\n", cameraEligibilitySuppressFrames);
	}

	int cameraEligibilityRestoreFrames_msg;
	if(n.getParam("cameraEligibilityRestoreFrames", cameraEligibilityRestoreFrames_msg))
	{
		cameraEligibilityRestoreFrames = cameraEligibilityRestoreFrames_msg;
		printf("camera eligibility restore frames %d!\n", cameraEligibilityRestoreFrames);
	}

	double cameraEligibilityScoreAlpha_msg;
	if(n.getParam("cameraEligibilityScoreAlpha", cameraEligibilityScoreAlpha_msg))
	{
		cameraEligibilityScoreAlpha = cameraEligibilityScoreAlpha_msg;
		printf("camera eligibility score alpha %f!\n", cameraEligibilityScoreAlpha);
	}

	int minFallbackWarmupFrames_msg;
	if(n.getParam("minFallbackWarmupFrames", minFallbackWarmupFrames_msg))
	{
		minFallbackWarmupFrames = minFallbackWarmupFrames_msg;
		printf("min fallback warmup frames %d!\n", minFallbackWarmupFrames);
	}

	int minFallbackKeyframes_msg;
	if(n.getParam("minFallbackKeyframes", minFallbackKeyframes_msg))
	{
		minFallbackKeyframes = minFallbackKeyframes_msg;
		printf("min fallback keyframes %d!\n", minFallbackKeyframes);
	}

	double maxFallbackReadyRMSE_msg;
	if(n.getParam("maxFallbackReadyRMSE", maxFallbackReadyRMSE_msg))
	{
		maxFallbackReadyRMSE = maxFallbackReadyRMSE_msg;
		printf("max fallback ready RMSE %f!\n", maxFallbackReadyRMSE);
	}

	int useRobustTrackingPolicy_msg;
	if(n.getParam("useRobustTrackingPolicy", useRobustTrackingPolicy_msg))
	{
		useRobustTrackingPolicy = useRobustTrackingPolicy_msg != 0;
		printf("robust tracking camera policy %s!\n", useRobustTrackingPolicy ? "enabled" : "disabled");
	}

	int trackingStatsWindowSize_msg;
	if(n.getParam("trackingStatsWindowSize", trackingStatsWindowSize_msg))
	{
		trackingStatsWindowSize = trackingStatsWindowSize_msg;
		printf("tracking stats window size %d!\n", trackingStatsWindowSize);
	}

	int minFallbackTrackingObservations_msg;
	if(n.getParam("minFallbackTrackingObservations", minFallbackTrackingObservations_msg))
	{
		minFallbackTrackingObservations = minFallbackTrackingObservations_msg;
		printf("min fallback tracking observations %d!\n", minFallbackTrackingObservations);
	}

	int minFallbackConsecutiveGoodFrames_msg;
	if(n.getParam("minFallbackConsecutiveGoodFrames", minFallbackConsecutiveGoodFrames_msg))
	{
		minFallbackConsecutiveGoodFrames = minFallbackConsecutiveGoodFrames_msg;
		printf("min fallback consecutive good frames %d!\n", minFallbackConsecutiveGoodFrames);
	}

	double minFallbackTrackingSuccessRate_msg;
	if(n.getParam("minFallbackTrackingSuccessRate", minFallbackTrackingSuccessRate_msg))
	{
		minFallbackTrackingSuccessRate = minFallbackTrackingSuccessRate_msg;
		printf("min fallback tracking success rate %f!\n", minFallbackTrackingSuccessRate);
	}

	double maxFallbackTrackingRmseStddev_msg;
	if(n.getParam("maxFallbackTrackingRmseStddev", maxFallbackTrackingRmseStddev_msg))
	{
		maxFallbackTrackingRmseStddev = maxFallbackTrackingRmseStddev_msg;
		printf("max fallback tracking RMSE stddev %f!\n", maxFallbackTrackingRmseStddev);
	}

	double maxFallbackPoseJumpRate_msg;
	if(n.getParam("maxFallbackPoseJumpRate", maxFallbackPoseJumpRate_msg))
	{
		maxFallbackPoseJumpRate = maxFallbackPoseJumpRate_msg;
		printf("max fallback pose jump rate %f!\n", maxFallbackPoseJumpRate);
	}

	double maxTrackingPoseJumpMeters_msg;
	if(n.getParam("maxTrackingPoseJumpMeters", maxTrackingPoseJumpMeters_msg))
	{
		maxTrackingPoseJumpMeters = maxTrackingPoseJumpMeters_msg;
		printf("max tracking pose jump %f meters!\n", maxTrackingPoseJumpMeters);
	}

	double trackingConfidenceScoreWeight_msg;
	if(n.getParam("trackingConfidenceScoreWeight", trackingConfidenceScoreWeight_msg))
	{
		trackingConfidenceScoreWeight = trackingConfidenceScoreWeight_msg;
		printf("tracking confidence score weight %f!\n", trackingConfidenceScoreWeight);
	}

	double trackingInstabilityPenaltyWeight_msg;
	if(n.getParam("trackingInstabilityPenaltyWeight", trackingInstabilityPenaltyWeight_msg))
	{
		trackingInstabilityPenaltyWeight = trackingInstabilityPenaltyWeight_msg;
		printf("tracking instability penalty weight %f!\n", trackingInstabilityPenaltyWeight);
	}

	int enableFallbackCameraRecovery_msg;
	if(n.getParam("enableFallbackCameraRecovery", enableFallbackCameraRecovery_msg))
	{
		enableFallbackCameraRecovery = enableFallbackCameraRecovery_msg != 0;
		printf("fallback camera recovery %s!\n", enableFallbackCameraRecovery ? "enabled" : "disabled");
	}

	int fallbackCameraRecoveryMinObservations_msg;
	if(n.getParam("fallbackCameraRecoveryMinObservations", fallbackCameraRecoveryMinObservations_msg))
	{
		fallbackCameraRecoveryMinObservations = fallbackCameraRecoveryMinObservations_msg;
		printf("fallback camera recovery min observations %d!\n", fallbackCameraRecoveryMinObservations);
	}

	int fallbackCameraRecoveryConsecutiveRejects_msg;
	if(n.getParam("fallbackCameraRecoveryConsecutiveRejects", fallbackCameraRecoveryConsecutiveRejects_msg))
	{
		fallbackCameraRecoveryConsecutiveRejects = fallbackCameraRecoveryConsecutiveRejects_msg;
		printf("fallback camera recovery consecutive rejects %d!\n", fallbackCameraRecoveryConsecutiveRejects);
	}

	double fallbackCameraRecoveryRejectedRate_msg;
	if(n.getParam("fallbackCameraRecoveryRejectedRate", fallbackCameraRecoveryRejectedRate_msg))
	{
		fallbackCameraRecoveryRejectedRate = fallbackCameraRecoveryRejectedRate_msg;
		printf("fallback camera recovery rejected rate %f!\n", fallbackCameraRecoveryRejectedRate);
	}

	double fallbackCameraRecoveryCooldown_msg;
	if(n.getParam("fallbackCameraRecoveryCooldown", fallbackCameraRecoveryCooldown_msg))
	{
		fallbackCameraRecoveryCooldown = fallbackCameraRecoveryCooldown_msg;
		printf("fallback camera recovery cooldown %f seconds!\n", fallbackCameraRecoveryCooldown);
	}

	double fallbackCameraQuarantineMinDuration_msg;
	if(n.getParam("fallbackCameraQuarantineMinDuration", fallbackCameraQuarantineMinDuration_msg))
	{
		fallbackCameraQuarantineMinDuration = fallbackCameraQuarantineMinDuration_msg;
		printf("fallback camera quarantine min duration %f seconds!\n", fallbackCameraQuarantineMinDuration);
	}

	int fallbackCameraRecoveryStableFrames_msg;
	if(n.getParam("fallbackCameraRecoveryStableFrames", fallbackCameraRecoveryStableFrames_msg))
	{
		fallbackCameraRecoveryStableFrames = fallbackCameraRecoveryStableFrames_msg;
		printf("fallback camera recovery stable frames %d!\n", fallbackCameraRecoveryStableFrames);
	}

	double fallbackCameraRecoveryMinImageStddev_msg;
	if(n.getParam("fallbackCameraRecoveryMinImageStddev", fallbackCameraRecoveryMinImageStddev_msg))
	{
		fallbackCameraRecoveryMinImageStddev = fallbackCameraRecoveryMinImageStddev_msg;
		printf("fallback camera recovery min image stddev %f!\n", fallbackCameraRecoveryMinImageStddev);
	}

	double fallbackCameraRecoveryMinGradientMean_msg;
	if(n.getParam("fallbackCameraRecoveryMinGradientMean", fallbackCameraRecoveryMinGradientMean_msg))
	{
		fallbackCameraRecoveryMinGradientMean = fallbackCameraRecoveryMinGradientMean_msg;
		printf("fallback camera recovery min gradient mean %f!\n", fallbackCameraRecoveryMinGradientMean);
	}

	int fallbackCameraRecoveryMinLidarPoints_msg;
	if(n.getParam("fallbackCameraRecoveryMinLidarPoints", fallbackCameraRecoveryMinLidarPoints_msg))
	{
		fallbackCameraRecoveryMinLidarPoints = fallbackCameraRecoveryMinLidarPoints_msg;
		printf("fallback camera recovery min lidar points %d!\n", fallbackCameraRecoveryMinLidarPoints);
	}

	double officialCameraMissingSwitchDelay_msg;
	if(n.getParam("officialCameraMissingSwitchDelay", officialCameraMissingSwitchDelay_msg))
	{
		officialCameraMissingSwitchDelay = officialCameraMissingSwitchDelay_msg;
		printf("official camera missing switch delay %f seconds!\n", officialCameraMissingSwitchDelay);
	}

	double maxFallbackCameraTimestampLag_msg;
	if(n.getParam("maxFallbackCameraTimestampLag", maxFallbackCameraTimestampLag_msg))
	{
		maxFallbackCameraTimestampLag = maxFallbackCameraTimestampLag_msg;
		printf("max fallback camera timestamp lag %f seconds!\n", maxFallbackCameraTimestampLag);
	}

	XmlRpc::XmlRpcValue cameraTimeOffsets_msg;
	if(n.getParam("cameraTimeOffsets", cameraTimeOffsets_msg))
	{
		cameraTimeOffsets.clear();
		if(cameraTimeOffsets_msg.getType() == XmlRpc::XmlRpcValue::TypeArray)
		{
			for(int i = 0; i < cameraTimeOffsets_msg.size(); ++i)
			{
				if(cameraTimeOffsets_msg[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
					cameraTimeOffsets.push_back(static_cast<double>(cameraTimeOffsets_msg[i]));
				else if(cameraTimeOffsets_msg[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
					cameraTimeOffsets.push_back(static_cast<int>(cameraTimeOffsets_msg[i]));
			}
		}
		else if(cameraTimeOffsets_msg.getType() == XmlRpc::XmlRpcValue::TypeDouble)
		{
			cameraTimeOffsets.push_back(static_cast<double>(cameraTimeOffsets_msg));
		}
		else if(cameraTimeOffsets_msg.getType() == XmlRpc::XmlRpcValue::TypeInt)
		{
			cameraTimeOffsets.push_back(static_cast<int>(cameraTimeOffsets_msg));
		}

		for(size_t i = 0; i < cameraTimeOffsets.size(); ++i)
			printf("camera %zu time offset %+f seconds!\n", i, cameraTimeOffsets[i]);
	}

	std::string imgTopic_msg;
	if(n.getParam("imgTopic", imgTopic_msg))
	{
		imgTopic = imgTopic_msg;
	}

	XmlRpc::XmlRpcValue imgTopics_msg;
	if(n.getParam("imgTopics", imgTopics_msg))
	{
		imgTopics.clear();
		if(imgTopics_msg.getType() == XmlRpc::XmlRpcValue::TypeArray)
		{
			for(int i = 0; i < imgTopics_msg.size(); ++i)
			{
				if(imgTopics_msg[i].getType() == XmlRpc::XmlRpcValue::TypeString)
					imgTopics.push_back(static_cast<std::string>(imgTopics_msg[i]));
			}
		}
		else if(imgTopics_msg.getType() == XmlRpc::XmlRpcValue::TypeString)
		{
			imgTopics.push_back(static_cast<std::string>(imgTopics_msg));
		}

		for(size_t i = 0; i < imgTopics.size(); ++i)
			printf("loading camera %zu images from topic %s!\n", i, imgTopics[i].c_str());
	}

	if(imgTopics.empty() && !imgTopic.empty())
	{
		imgTopics.push_back(imgTopic);
		printf("loading images from topic %s!\n", imgTopic.c_str());
	}

	std::string lidarTopic_msg;
	if(n.getParam("lidarTopic", lidarTopic_msg))
	{
		lidarTopic = lidarTopic_msg;
		printf("loading images from topic %s!\n", lidarTopic.c_str());
	}

	XmlRpc::XmlRpcValue calib_msg;
	if(n.getParam("calib", calib_msg))
	{
		calibPaths.clear();
		if(calib_msg.getType() == XmlRpc::XmlRpcValue::TypeString)
		{
			calib = resolveRosFindSubstitutions(static_cast<std::string>(calib_msg));
			calibPaths.push_back(calib);
		}
		else if(calib_msg.getType() == XmlRpc::XmlRpcValue::TypeArray && calib_msg.size() > 0)
		{
			for(int i = 0; i < calib_msg.size(); ++i)
			{
				if(calib_msg[i].getType() == XmlRpc::XmlRpcValue::TypeString)
					calibPaths.push_back(resolveRosFindSubstitutions(static_cast<std::string>(calib_msg[i])));
			}

			if(!calibPaths.empty())
				calib = calibPaths[0];
		}

		for(size_t i = 0; i < calibPaths.size(); ++i)
			printf("loading camera %zu calibration from %s!\n", i, calibPaths[i].c_str());
	}

	std::string pathSensorPrameter_msg;
	if(n.getParam("pathSensorPrameter", pathSensorPrameter_msg))
	{
		pathSensorPrameter = resolveRosFindSubstitutions(pathSensorPrameter_msg);
	}

	XmlRpc::XmlRpcValue pathSensorPrameters_msg;
	if(n.getParam("pathSensorPrameters", pathSensorPrameters_msg))
	{
		pathSensorPrameters.clear();
		if(pathSensorPrameters_msg.getType() == XmlRpc::XmlRpcValue::TypeArray)
		{
			for(int i = 0; i < pathSensorPrameters_msg.size(); ++i)
			{
				if(pathSensorPrameters_msg[i].getType() == XmlRpc::XmlRpcValue::TypeString)
					pathSensorPrameters.push_back(resolveRosFindSubstitutions(static_cast<std::string>(pathSensorPrameters_msg[i])));
			}
		}
		else if(pathSensorPrameters_msg.getType() == XmlRpc::XmlRpcValue::TypeString)
		{
			pathSensorPrameters.push_back(resolveRosFindSubstitutions(static_cast<std::string>(pathSensorPrameters_msg)));
		}

		for(size_t i = 0; i < pathSensorPrameters.size(); ++i)
			printf("loading camera %zu sensor parameters from %s!\n", i, pathSensorPrameters[i].c_str());
	}
	else if(!pathSensorPrameter.empty())
	{
		printf("loading the prameters of camera and lidar from %s!\n", pathSensorPrameter.c_str());
	}

	std::string resultPath_msg;
	if(n.getParam("resultPath", resultPath_msg))
	{
		resultPath = resultPath_msg;
		printf("save result at %s!\n", resultPath.c_str());
	}

	std::string cameraScoreLogPath_msg;
	if(n.getParam("cameraScoreLogPath", cameraScoreLogPath_msg))
	{
		cameraScoreLogPath = cameraScoreLogPath_msg;
		printf("save camera score log at %s!\n", cameraScoreLogPath.c_str());
	}

    std::string vignette_msg;
	if(n.getParam("vignette", vignette_msg))
	{
		vignette = vignette_msg;
		printf("loading vignette from %s!\n", vignette.c_str());
	}

    std::string gamma_msg;
	if(n.getParam("gamma", gamma_msg))
	{
		gammaCalib = gamma_msg;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
	}

    float rescale_msg;
	if(n.getParam("rescale", rescale_msg))
	{
		rescale = rescale_msg;
		printf("RESCALE %f!\n", rescale);
	}

    float speed_msg;
	if(n.getParam("speed", speed_msg))
	{
		playbackSpeed = speed_msg;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
	}

    int save_msg;
	if(n.getParam("save", save_msg))
	{
		if(save_msg==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

    int mode_msg;
	if(n.getParam("mode", mode_msg))
	{

		mode = mode_msg;
		if(mode_msg==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(mode_msg==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0;
			setting_affineOptModeB = 0;
		}
		if(mode_msg==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1;
			setting_affineOptModeB = -1;
            setting_minGradHistAdd=3;
		}
	}
}

void process()
{
	if(!hasSynchronizedCameraCandidates())
    {
        printNoSyncDiagnostics();
		return;
    }

    std::vector<sdv_loam::CameraFrameInput> candidates = buildCurrentCameraCandidates();
    std::vector<int> invalidQualityCameraIds;
    std::set<int> invalidQualityCameraSet;
    static unsigned long long lowTextureDropCount = 0;
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const sdv_loam::CameraFrameInput& candidate = candidates[i];
        const sdv_loam::CameraSelectionScore score =
            sdv_loam::scoreCameraCandidate(
                candidate,
                fullSystem->getVisualState(candidate.cameraId),
                maxCameraLidarSyncError,
                minCameraImageStddev,
                minCameraImageGradientMean,
                minCameraImageLaplacianVariance);

        if(candidate.cameraId == candidate.lidarProjection.cameraId &&
           fabs(candidate.imageTimestamp - candidate.lidarProjection.lidarTimestamp) <= maxCameraLidarSyncError &&
           !fullSystem->getVisualState(candidate.cameraId).initFailed &&
           !score.valid &&
           (score.imageStddev < minCameraImageStddev ||
            score.imageGradientMean < minCameraImageGradientMean ||
            score.imageLaplacianVariance < minCameraImageLaplacianVariance))
        {
            invalidQualityCameraIds.push_back(candidate.cameraId);
            invalidQualityCameraSet.insert(candidate.cameraId);
            lowTextureDropCount++;
            if(lowTextureDropCount % 30 == 1)
            {
                printf("dropping degraded camera candidate camera=%d time=%.6f mean=%.3f stddev=%.3f grad=%.3f lap_var=%.3f thresholds: stddev>=%.3f grad>=%.3f lap_var>=%.3f\n",
                       candidate.cameraId,
                       candidate.imageTimestamp,
                       score.imageMean,
                       score.imageStddev,
                       score.imageGradientMean,
                       score.imageLaplacianVariance,
                       minCameraImageStddev,
                       minCameraImageGradientMean,
                       minCameraImageLaplacianVariance);
            }
        }
    }

    std::map<int, const sdv_loam::CameraFrameInput*> validCandidatesByCamera;
    std::map<int, sdv_loam::CameraSelectionScore> allScoresByCamera;
    std::map<int, sdv_loam::CameraSelectionScore> validScoresByCamera;
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const sdv_loam::CameraFrameInput& candidate = candidates[i];
        sdv_loam::CameraSelectionScore score =
            sdv_loam::scoreCameraCandidate(
                candidate,
                fullSystem->getVisualState(candidate.cameraId),
                maxCameraLidarSyncError,
                minCameraImageStddev,
                minCameraImageGradientMean,
                minCameraImageLaplacianVariance);

        if(score.valid)
        {
            applyTrackingStatsToScore(candidate.cameraId, score);
            validCandidatesByCamera[candidate.cameraId] = &candidate;
            validScoresByCamera[candidate.cameraId] = score;
        }
        allScoresByCamera[candidate.cameraId] = score;
    }

    updateCameraEligibility(candidates, validScoresByCamera);

    const sdv_loam::CameraFrameInput* selectedInput = 0;
    if(officialCameraId < 0)
    {
        if(trackingCameraId >= 0 && validCandidatesByCamera.count(trackingCameraId) > 0)
        {
            officialCameraId = trackingCameraId;
            selectedInput = validCandidatesByCamera[officialCameraId];
            printf("official tracking camera initialized with camera %d.\n", officialCameraId);
        }
        else
        {
            double bestScore = -std::numeric_limits<double>::infinity();
            for(std::map<int, sdv_loam::CameraSelectionScore>::const_iterator it = validScoresByCamera.begin();
                it != validScoresByCamera.end();
                ++it)
            {
                if(it->second.score > bestScore)
                {
                    bestScore = it->second.score;
                    officialCameraId = it->first;
                    selectedInput = validCandidatesByCamera[it->first];
                }
            }
            if(selectedInput != 0)
                printf("official tracking camera initialized with camera %d.\n", officialCameraId);
        }
    }
    else
    {
        double latestCandidateTimestamp = -1.0;
        for(size_t i = 0; i < candidates.size(); ++i)
            latestCandidateTimestamp = std::max(latestCandidateTimestamp, candidates[i].imageTimestamp);

        const sdv_loam::VisualState& officialState = fullSystem->getVisualState(officialCameraId);
        const bool officialValid = validCandidatesByCamera.count(officialCameraId) > 0;
        const bool officialLowTexture = invalidQualityCameraSet.count(officialCameraId) > 0;
        const bool officialStateFailed = officialState.isLost || officialState.initFailed;
        const bool officialRmseBad =
            std::isfinite(officialState.lastCoarseRMSE[0]) &&
            officialState.lastCoarseRMSE[0] > maxCoarseTrackingRMSE;
        const bool officialSuppressed = !isCameraEligible(officialCameraId);
        const bool officialTrackingUnstable =
            useRobustTrackingPolicy &&
            cameraTrackingStatsByCamera.count(officialCameraId) > 0 &&
            cameraTrackingStatsByCamera[officialCameraId].observations >= std::max(1, minFallbackTrackingObservations) &&
            !isCameraTrackingStable(officialCameraId);
        const bool officialMissingLongEnough =
            !officialValid &&
            lastOfficialFrameTimestamp >= 0.0 &&
            latestCandidateTimestamp >= 0.0 &&
            latestCandidateTimestamp - lastOfficialFrameTimestamp > officialCameraMissingSwitchDelay;
        const bool emergencyOfficialSwitch = officialStateFailed || officialMissingLongEnough;
        const bool degradedOfficial = officialLowTexture || officialRmseBad || officialSuppressed || officialTrackingUnstable;

        if(officialValid)
            selectedInput = validCandidatesByCamera[officialCameraId];

        double bestScore = -std::numeric_limits<double>::infinity();
        int fallbackCameraId = -1;
        for(std::map<int, sdv_loam::CameraSelectionScore>::const_iterator it = validScoresByCamera.begin();
            it != validScoresByCamera.end();
            ++it)
        {
            const int cameraId = it->first;
            if(cameraId == officialCameraId)
                continue;
            if(!isCameraEligible(cameraId))
                continue;

            const sdv_loam::CameraFrameInput* fallbackCandidate = validCandidatesByCamera[cameraId];
            const double fallbackTimestampLag =
                latestCandidateTimestamp >= 0.0 && fallbackCandidate != 0
                    ? latestCandidateTimestamp - fallbackCandidate->imageTimestamp
                    : 0.0;
            const bool fallbackTimeAligned =
                maxFallbackCameraTimestampLag < 0.0 ||
                fallbackTimestampLag <= maxFallbackCameraTimestampLag;
            if(!fallbackTimeAligned)
                continue;
            if(isFallbackCameraQuarantined(cameraId, latestCandidateTimestamp))
                continue;

            std::map<int, double>::const_iterator resetIt = lastCameraResetTimestampByCamera.find(cameraId);
            const bool recentlyReset =
                cameraResetSwitchCooldown >= 0.0 &&
                resetIt != lastCameraResetTimestampByCamera.end() &&
                latestCandidateTimestamp >= 0.0 &&
                latestCandidateTimestamp - resetIt->second < cameraResetSwitchCooldown;
            if(recentlyReset)
                continue;

            const sdv_loam::VisualState& state = fullSystem->getVisualState(cameraId);
            const bool fallbackReady =
                state.initialized &&
                !state.isLost &&
                !state.initFailed &&
                std::isfinite(state.lastCoarseRMSE[0]) &&
                state.lastCoarseRMSE[0] <= maxFallbackReadyRMSE &&
                static_cast<int>(state.allFrameHistory.size()) >= minFallbackWarmupFrames &&
                static_cast<int>(state.allKeyFramesHistory.size()) >= minFallbackKeyframes &&
                isCameraTrackingStable(cameraId);

            if(!fallbackReady)
                continue;

            if(it->second.score > bestScore)
            {
                bestScore = it->second.score;
                fallbackCameraId = cameraId;
            }
        }

        bool shouldSwitch = false;
        const bool switchIntervalReady =
            cameraSwitchMinInterval < 0.0 ||
            lastCameraSwitchTimestamp < 0.0 ||
            latestCandidateTimestamp < 0.0 ||
            latestCandidateTimestamp - lastCameraSwitchTimestamp >= cameraSwitchMinInterval;
        const double officialScore =
            officialValid && validScoresByCamera.count(officialCameraId) > 0
                ? validScoresByCamera[officialCameraId].score
                : -std::numeric_limits<double>::infinity();
        const bool betterByScore =
            fallbackCameraId >= 0 &&
            bestScore > officialScore + cameraSwitchScoreMargin;
        const bool betterAlternative =
            fallbackCameraId >= 0 &&
            switchIntervalReady &&
            (betterByScore || emergencyOfficialSwitch || degradedOfficial || !officialValid);

        if(betterAlternative)
        {
            const bool immediateSwitch = officialStateFailed || officialMissingLongEnough;
            if(immediateSwitch)
            {
                shouldSwitch = true;
                pendingSwitchCameraId = -1;
                pendingSwitchCount = 0;
            }
            else
            {
                if(pendingSwitchCameraId == fallbackCameraId)
                    pendingSwitchCount++;
                else
                {
                    pendingSwitchCameraId = fallbackCameraId;
                    pendingSwitchCount = 1;
                }

                shouldSwitch = pendingSwitchCount >= std::max(1, cameraSwitchConsecutiveFrames);
            }
        }
        else
        {
            pendingSwitchCameraId = -1;
            pendingSwitchCount = 0;
        }

        if(shouldSwitch && fallbackCameraId >= 0)
        {
            const double fallbackTimestamp =
                validCandidatesByCamera.count(fallbackCameraId) > 0
                    ? validCandidatesByCamera[fallbackCameraId]->imageTimestamp
                    : -1.0;
            const char* switchReason =
                officialStateFailed ? "state_failed" :
                officialMissingLongEnough ? "missing" :
                officialSuppressed ? "suppressed" :
                degradedOfficial ? "degraded" :
                betterByScore ? "better_score" :
                !officialValid ? "unavailable" :
                "unknown";
            printf("official tracking camera switch %d -> %d (reason=%s emergency=%d degraded=%d better_score=%d low_texture=%d state_failed=%d rmse_bad=%d suppressed=%d tracking_unstable=%d missing=%.3f s active_score=%.1f fallback_score=%.1f fallback_lag=%.3f s pending=%d/%d).\n",
                   officialCameraId,
                   fallbackCameraId,
                   switchReason,
                   emergencyOfficialSwitch ? 1 : 0,
                   degradedOfficial ? 1 : 0,
                   betterByScore ? 1 : 0,
                   officialLowTexture ? 1 : 0,
                   officialStateFailed ? 1 : 0,
                   officialRmseBad ? 1 : 0,
                   officialSuppressed ? 1 : 0,
                   officialTrackingUnstable ? 1 : 0,
                   lastOfficialFrameTimestamp >= 0.0 && latestCandidateTimestamp >= 0.0 ? latestCandidateTimestamp - lastOfficialFrameTimestamp : -1.0,
                   officialScore,
                   bestScore,
                   latestCandidateTimestamp >= 0.0 && fallbackTimestamp >= 0.0 ? latestCandidateTimestamp - fallbackTimestamp : -1.0,
                   pendingSwitchCount,
                   std::max(1, cameraSwitchConsecutiveFrames));
            const SE3 preservedRigPose = fullSystem->getRigPose();
            const bool alignedFallback = fullSystem->alignVisualStateToRigPose(fallbackCameraId, preservedRigPose);
            if(!alignedFallback)
            {
                printf("warning: rejected camera switch %d -> %d because fallback state could not be aligned to preserved rig pose.\n",
                       officialCameraId,
                       fallbackCameraId);
            }
            else
            {
                officialCameraId = fallbackCameraId;
                selectedInput = validCandidatesByCamera[fallbackCameraId];
                lastCameraSwitchTimestamp = latestCandidateTimestamp;
                pendingSwitchCameraId = -1;
                pendingSwitchCount = 0;
                pendingPostSwitchRigAlignment = true;
                pendingPostSwitchCameraId = fallbackCameraId;
                pendingPostSwitchRigPose = preservedRigPose;
            }
        }
    }

    double latestCandidateTimestampForLog = -1.0;
    for(size_t i = 0; i < candidates.size(); ++i)
        latestCandidateTimestampForLog = std::max(latestCandidateTimestampForLog, candidates[i].imageTimestamp);
    writeCameraScoreLog(
        candidates,
        allScoresByCamera,
        invalidQualityCameraSet,
        selectedInput != 0 ? selectedInput->cameraId : -1,
        latestCandidateTimestampForLog);

    if(selectedInput == 0)
    {
        double latestCandidateTimestamp = -1.0;
        for(size_t i = 0; i < candidates.size(); ++i)
            latestCandidateTimestamp = std::max(latestCandidateTimestamp, candidates[i].imageTimestamp);
        dropStaleCameraCandidatesExcept(-1, latestCandidateTimestamp);

        for(size_t i = 0; i < candidates.size(); ++i)
        {
            const sdv_loam::CameraFrameInput& warmupInput = candidates[i];
            if(invalidQualityCameraSet.count(warmupInput.cameraId) > 0)
                continue;
            if(!isCameraEligible(warmupInput.cameraId))
                continue;
            if(validCandidatesByCamera.count(warmupInput.cameraId) == 0)
                continue;
            if(maxFallbackCameraTimestampLag >= 0.0 &&
               latestCandidateTimestamp - warmupInput.imageTimestamp > maxFallbackCameraTimestampLag)
                continue;

            applyVisualCalibrationForCamera(warmupInput.cameraId);
            ImageFolderReader* warmupReader = getReaderForCamera(warmupInput.cameraId);
            cv::Mat warmupFrame = warmupInput.frame;
            ImageAndExposure* warmupImg = warmupReader->getRosImage(warmupFrame, warmupInput.imageTimestamp);
            const double warmupScore = validScoresByCamera.count(warmupInput.cameraId) > 0
                ? validScoresByCamera[warmupInput.cameraId].score
                : std::numeric_limits<double>::quiet_NaN();
            const size_t warmupHistorySizeBefore =
                fullSystem->getVisualState(warmupInput.cameraId).allFrameHistory.size();
            fullSystem->addActiveFrame(
                sdv_loam::CameraInput(
                    warmupInput.cameraId,
                    warmupImg,
                    currentId,
                    warmupInput.lidarProjection,
                    warmupScore),
                false);
            updateCameraTrackingStats(warmupInput.cameraId, warmupHistorySizeBefore);
            std::map<int, sdv_loam::CameraSelectionScore>::const_iterator warmupScoreIt =
                validScoresByCamera.find(warmupInput.cameraId);
            if(warmupScoreIt != validScoresByCamera.end())
                updateFallbackCameraRecoveryState(
                    warmupInput.cameraId,
                    warmupScoreIt->second,
                    warmupInput.imageTimestamp);
            delete warmupImg;
        }

        for(size_t i = 0; i < candidates.size(); ++i)
            if(fullSystem->hasQueuedCameraCandidate(candidates[i].cameraId))
                fullSystem->popQueuedCameraCandidate(candidates[i].cameraId);
        return;
    }

    const sdv_loam::CameraFrameInput& input = *selectedInput;
    std::set<int> cameraIdsToConsume;
    cameraIdsToConsume.insert(input.cameraId);
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const sdv_loam::CameraFrameInput& candidate = candidates[i];
        const bool sameLidarCycle =
            fabs(candidate.lidarProjection.lidarTimestamp - input.lidarProjection.lidarTimestamp) <= maxCameraLidarSyncError;
        const bool sameImageCycle =
            fabs(candidate.imageTimestamp - input.imageTimestamp) <= maxCameraLidarSyncError;

        if(sameLidarCycle || sameImageCycle || invalidQualityCameraSet.count(candidate.cameraId) > 0)
            cameraIdsToConsume.insert(candidate.cameraId);
    }

    timestamp = input.imageTimestamp;

    if(currentId == 0)
    	initialtimestamp = timestamp;

    sdv_loam::VisualState& activeVisualState = fullSystem->getVisualState(input.cameraId);

    if(!activeVisualState.initialized)
    {
        gettimeofday(&tv_start, NULL);
        started = clock();
        sInitializerOffset = timestamp - initialtimestamp;
    }

    if(firstFlag)
    {
        firstFrameTime = timestamp;
        firstFlag = false;
    }

    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const sdv_loam::CameraFrameInput& warmupInput = candidates[i];
        if(warmupInput.cameraId == input.cameraId)
            continue;
        if(invalidQualityCameraSet.count(warmupInput.cameraId) > 0)
            continue;
        if(!isCameraEligible(warmupInput.cameraId))
            continue;

        const bool sameLidarCycle =
            fabs(warmupInput.lidarProjection.lidarTimestamp - input.lidarProjection.lidarTimestamp) <= maxCameraLidarSyncError;
        const bool sameImageCycle =
            fabs(warmupInput.imageTimestamp - input.imageTimestamp) <= maxCameraLidarSyncError;
        if(!sameLidarCycle && !sameImageCycle)
            continue;

        applyVisualCalibrationForCamera(warmupInput.cameraId);
        ImageFolderReader* warmupReader = getReaderForCamera(warmupInput.cameraId);
        cv::Mat warmupFrame = warmupInput.frame;
        ImageAndExposure* warmupImg = warmupReader->getRosImage(warmupFrame, warmupInput.imageTimestamp);
        const double warmupScore = validScoresByCamera.count(warmupInput.cameraId) > 0
            ? validScoresByCamera[warmupInput.cameraId].score
            : std::numeric_limits<double>::quiet_NaN();
        const size_t warmupHistorySizeBefore =
            fullSystem->getVisualState(warmupInput.cameraId).allFrameHistory.size();
        fullSystem->addActiveFrame(
            sdv_loam::CameraInput(
                warmupInput.cameraId,
                warmupImg,
                currentId,
                warmupInput.lidarProjection,
                warmupScore),
            false);
        updateCameraTrackingStats(warmupInput.cameraId, warmupHistorySizeBefore);
        std::map<int, sdv_loam::CameraSelectionScore>::const_iterator warmupScoreIt =
            validScoresByCamera.find(warmupInput.cameraId);
        if(warmupScoreIt != validScoresByCamera.end())
            updateFallbackCameraRecoveryState(
                warmupInput.cameraId,
                warmupScoreIt->second,
                warmupInput.imageTimestamp);
        delete warmupImg;
    }

    applyVisualCalibrationForCamera(input.cameraId);

    ImageAndExposure* img;
    cv::Mat selectedFrame = input.frame;
    ImageFolderReader* cameraReader = getReaderForCamera(input.cameraId);
    img = cameraReader->getRosImage(selectedFrame, input.imageTimestamp);

    bool skipFrame=false;

    const double inputScore = validScoresByCamera.count(input.cameraId) > 0
        ? validScoresByCamera[input.cameraId].score
        : std::numeric_limits<double>::quiet_NaN();
    if(!skipFrame)
    {
        const size_t inputHistorySizeBefore =
            fullSystem->getVisualState(input.cameraId).allFrameHistory.size();
        fullSystem->addActiveFrame(
            sdv_loam::CameraInput(
                input.cameraId,
                img,
                currentId,
                input.lidarProjection,
                inputScore));
        updateCameraTrackingStats(input.cameraId, inputHistorySizeBefore);
        if(pendingPostSwitchRigAlignment && input.cameraId == pendingPostSwitchCameraId)
        {
            const SE3 trackedRigPose = fullSystem->getRigPose();
            const double trackerJump =
                (trackedRigPose.translation() - pendingPostSwitchRigPose.translation()).norm();
            const bool realignedPostSwitch =
                fullSystem->alignVisualStateToRigPose(input.cameraId, pendingPostSwitchRigPose);
            if(realignedPostSwitch)
            {
                printf("post-switch rig pose realigned camera=%d tracker_jump=%.3f m.\n",
                       input.cameraId,
                       trackerJump);
            }
            else
            {
                printf("warning: post-switch rig pose realignment failed camera=%d tracker_jump=%.3f m.\n",
                       input.cameraId,
                       trackerJump);
            }

            pendingPostSwitchRigAlignment = false;
            pendingPostSwitchCameraId = -1;
        }
        fullSystem->recordRigPoseSnapshot(input.lidarProjection.lidarTimestamp, input.cameraId);
    }
    lastOfficialFrameTimestamp = input.imageTimestamp;

    currentId++;

    delete img;
    for(std::set<int>::const_iterator it = cameraIdsToConsume.begin(); it != cameraIdsToConsume.end(); ++it)
    {
        if(fullSystem->hasQueuedCameraCandidate(*it))
            fullSystem->popQueuedCameraCandidate(*it);
    }
    dropStaleCameraCandidatesExcept(input.cameraId, input.imageTimestamp);

    if(currentId % kResultSaveIntervalFrames == 0 && !resultPath.empty())
        fullSystem->printResult(resultPath);

    bool activeInitFailed = activeVisualState.initFailed;
    bool activeIsLost = activeVisualState.isLost;

    if(activeInitFailed || setting_fullResetRequested)
    {
        if(currentId < 250 || setting_fullResetRequested)
        {
            printf("RESETTING!\n");

            std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
            delete fullSystem;

            for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

            fullSystem = new FullSystem();
            fullSystem->setGammaFunction(reader->getPhotometricGamma());
            fullSystem->linearizeOperation = (playbackSpeed==0);
            applyRuntimeFullSystemSettings();
            configureImageCameras();
            loadCameraSensorParameters();
            applyAllCameraIntrinsics();
            fullSystem->outputWrapper = wraps;
            officialCameraId = -1;
            lastOfficialFrameTimestamp = -1.0;
            lastCameraSwitchTimestamp = -1.0;
            pendingSwitchCameraId = -1;
            pendingSwitchCount = 0;
            lastCameraResetTimestampByCamera.clear();
            quarantinedFallbackCameraSinceByCamera.clear();
            fallbackCameraRecoveryGoodFramesByCamera.clear();
            cameraEligibilityStatsByCamera.clear();
            cameraTrackingStatsByCamera.clear();
            eligibleCameraSet.clear();

            setting_fullResetRequested=false;
            activeIsLost = false;
        }
    }

    if(activeIsLost)
    {
        printf("LOST!!\n");
        return;
    }
}

void imgHandlerForCamera(const sensor_msgs::ImageConstPtr &img_msg, int cameraId)
{
    receivedImagesByCamera[cameraId]++;
    lastReceivedImageTimestampByCamera[cameraId] = img_msg->header.stamp.toSec();

    if(!acceptsCamera(cameraId))
        return;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat currentFrame = ptr->image;

    double timeImg = img_msg->header.stamp.toSec() + getCameraTimeOffset(cameraId);
    lastEnqueuedImageTimestampByCamera[cameraId] = timeImg;
    fullSystem->enqueueCameraFrame(cameraId, currentFrame, timeImg);
}

void imgHandler(const sensor_msgs::ImageConstPtr &img_msg)
{
    imgHandlerForCamera(img_msg, kDefaultCameraId);
}

void projectPointCloud()
{
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize; 
    PointType thisPoint;

    cloudSize = laserCloudIn->points.size();

    for (size_t i = 0; i < cloudSize; ++i){

        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;

        laserCloudIn->points[i].intensity = -1;

        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
        if (rowIdn < 0 || rowIdn >= N_SCAN)
            continue;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (columnIdn >= Horizon_SCAN)
            columnIdn -= Horizon_SCAN;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;

        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        if (range < 0.1)
            continue;

        rangeMat.at<float>(rowIdn, columnIdn) = range;

        thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn  + rowIdn * Horizon_SCAN;
        fullCloud->points[index] = thisPoint;
        fullInfoCloud->points[index] = thisPoint;
        fullInfoCloud->points[index].intensity = range;
        laserCloudIn->points[i].intensity = range;
    }
}

void groundRemoval()
{
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> candidateGround;

    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;

    for (size_t j = 0; j < Horizon_SCAN; ++j){
        for (size_t i = 0; i < groundScanInd; ++i){

            lowerInd = j + ( i )*Horizon_SCAN;
            upperInd = j + (i+1)*Horizon_SCAN;

            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1)
            {
                groundMat.at<int8_t>(i,j) = -1;
                continue;
            }
                
            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

            if (abs(angle - sensorMountAngle) <= 10)
            {
                groundMat.at<int8_t>(i,j) = 1;
                groundMat.at<int8_t>(i+1,j) = 1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                labelMat.at<int>(i,j) = -1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1)
                groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
        }
    }
}

void labelComponents(int row, int col)
{
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY; 
    bool lineCountFlag[N_SCAN] = {false};

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;
    
    while(queueSize > 0)
    {
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;
        ++queueStartInd;

        labelMat.at<int>(fromIndX, fromIndY) = labelCount;

        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
        {
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;

            if (thisIndX < 0 || thisIndX >= N_SCAN)
                continue;

            if (thisIndY < 0)
                thisIndY = Horizon_SCAN - 1;
            if (thisIndY >= Horizon_SCAN)
                thisIndY = 0;

            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                          rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                          rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segmentAlphaX;
            else
                alpha = segmentAlphaY;

            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

            if (angle > segmentTheta){

                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    bool feasibleSegment = false;

    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum){
        int lineCount = 0;
        for (size_t i = 0; i < N_SCAN; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;
        if (lineCount >= segmentValidLineNum)
            feasibleSegment = true;            
    }

    if (feasibleSegment == true){
        ++labelCount;
    }else{
        for (size_t i = 0; i < allPushedIndSize; ++i){
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

void cloudSegmentation()
{
    for (size_t i = 0; i < N_SCAN; ++i)
        for (size_t j = 0; j < Horizon_SCAN; ++j)
            if (labelMat.at<int>(i,j) == 0)
                labelComponents(i, j);


    int sizeOfSegCloud = 0;

    for (size_t i = 0; i < N_SCAN; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                if (labelMat.at<int>(i,j) == 999999){
                    if (i > groundScanInd && j % 5 == 0){
                        outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        continue;
                    }else{
                        continue;
                    }
                }
                
                if(groundMat.at<int8_t>(i,j) == 1)
                	fullCloud->points[j + i*Horizon_SCAN].intensity = -1.0;
                else
                	fullCloud->points[j + i*Horizon_SCAN].intensity = 1.0;

                segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);

                ++sizeOfSegCloud;
            }
        }
    }
}

void lidarCloudHandler(const sensor_msgs::PointCloud2ConstPtr& lidarCloudMsg){
    double timeLidarCloud = lidarCloudMsg->header.stamp.toSec();

    pcl::fromROSMsg(*lidarCloudMsg, *laserCloudIn);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

    projectPointCloud();

    groundRemoval();

    cloudSegmentation();

    int cloudSize = segmentedCloud->points.size();

    const std::vector<int> cameraIds = fullSystem->getConfiguredCameraIds();
    bool addFeaturePointForAnyCamera = false;

    for(size_t cameraIdx = 0; cameraIdx < cameraIds.size(); ++cameraIdx)
    {
        const int cameraId = cameraIds[cameraIdx];
        if(!acceptsCamera(cameraId))
            continue;

        const sdv_loam::CameraCalibration& cameraCalibration = fullSystem->getCameraCalibration(cameraId);
        const Eigen::Matrix3d Rlc = cameraCalibration.T_LC.rotationMatrix();
        const Eigen::Vector3d tlc = cameraCalibration.T_LC.translation();

        Eigen::Vector3d lidarCloudTemp;
        Eigen::Vector3d cloudPixelTemp;
        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel;
        sdv_loam::LidarProjectionResult projectionResult;
        projectionResult.cameraId = cameraId;
        projectionResult.lidarTimestamp = timeLidarCloud;

        int numGround = 0;
        int numAll = 0;

        for (size_t i = 0; i < cloudSize; ++i){
            lidarCloudTemp(0, 0) = segmentedCloud->points[i].x;
            lidarCloudTemp(1, 0) = segmentedCloud->points[i].y;
            lidarCloudTemp(2, 0) = segmentedCloud->points[i].z;

            Eigen::Vector3d temp = Rlc * lidarCloudTemp + tlc;

            if(temp(2, 0) < 0.2)
            {
                continue;
            }

            float u = (float)(temp(0, 0) / temp(2, 0));
            float v = (float)(temp(1, 0) / temp(2, 0));

            float Ku = u * cameraCalibration.fx + cameraCalibration.cx;
            float Kv = v * cameraCalibration.fy + cameraCalibration.cy;

            if((int)Ku<4 || (int)Ku>=wG[0]-5 || (int)Kv<4 || (int)Kv>hG[0]-4)
            {
                continue;
            }

            cloudPixelTemp(0, 0) = (double)Ku;
            cloudPixelTemp(1, 0) = (double)Kv;
            cloudPixelTemp(2, 0) = temp(2, 0);

            numAll++;

            if(segmentedCloud->points[i].intensity < 0)
                numGround++;

            vCloudPixel.push_back(cloudPixelTemp);
        }

        if(numAll > 0 && float(numGround)/(float)numAll > 0.8)
            addFeaturePointForAnyCamera = true;

        projectionResult.cloudPixels = vCloudPixel;
        receivedLidarProjectionsByCamera[cameraId]++;
        lastReceivedLidarTimestampByCamera[cameraId] = timeLidarCloud;
        lastEnqueuedLidarTimestampByCamera[cameraId] = projectionResult.lidarTimestamp;
        fullSystem->enqueueLidarProjection(projectionResult);
    }

    fullSystem->addFeaturePoint = addFeaturePointForAnyCamera;

    resetParameters();
}

void lidarPoseMapHandler(const nav_msgs::Odometry::ConstPtr& lidarPoseMapMsg){
    double timeLidarPoseMap = lidarPoseMapMsg->header.stamp.toSec();
    fullSystem->qTimeLidarPoseMap.push(timeLidarPoseMap);

    Eigen::Quaterniond q;
	q.w()=lidarPoseMapMsg->pose.pose.orientation.w;
	q.x()=lidarPoseMapMsg->pose.pose.orientation.x;
	q.y()=lidarPoseMapMsg->pose.pose.orientation.y;
	q.z()=lidarPoseMapMsg->pose.pose.orientation.z;

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t(lidarPoseMapMsg->pose.pose.position.x, lidarPoseMapMsg->pose.pose.position.y, lidarPoseMapMsg->pose.pose.position.z);

	fullSystem->qRotationMap.push(R);
	fullSystem->qPositionMap.push(t);
}

void lidarPoseOdometerHandler(const nav_msgs::Odometry::ConstPtr& lidarPoseOdometerMsg){
    double timeLidarPoseOdometer = lidarPoseOdometerMsg->header.stamp.toSec();
    fullSystem->qTimeLidarPoseOdometer.push(timeLidarPoseOdometer);

    Eigen::Quaterniond q;
	q.w()=lidarPoseOdometerMsg->pose.pose.orientation.w;
	q.x()=lidarPoseOdometerMsg->pose.pose.orientation.x;
	q.y()=lidarPoseOdometerMsg->pose.pose.orientation.y;
	q.z()=lidarPoseOdometerMsg->pose.pose.orientation.z;

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t(lidarPoseOdometerMsg->pose.pose.position.x, lidarPoseOdometerMsg->pose.pose.position.y, lidarPoseOdometerMsg->pose.pose.position.z);

	fullSystem->qRotationOdometer.push(R);
	fullSystem->qPositionOdometer.push(t);
}

int main(int argc, char** argv)
{
    signal(SIGSEGV, segfaultHandler);
    signal(SIGABRT, segfaultHandler);

	ros::init(argc, argv, "sdv_loam");

	ROS_INFO("\033[1;32m---->\033[SDV-LOAM Started.");

	ros::NodeHandle n;

    parseArgument(n);

    allocateMemory();
    resetParameters();

    createCameraReaders();
    if(reader == NULL)
        reader = new ImageFolderReader(calib, gammaCalib, vignette);
	reader->setGlobalCalibration();

    fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);
	applyRuntimeFullSystemSettings();
	configureImageCameras();
	loadCameraSensorParameters();
	applyAllCameraIntrinsics();

	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    std::thread runthread([&]() {
        gettimeofday(&tv_start, NULL);
        std::vector<ros::Subscriber> sub_imgs;
        sub_imgs.reserve(imgTopics.size());
        for(size_t cameraId = 0; cameraId < imgTopics.size(); ++cameraId)
        {
            sub_imgs.push_back(n.subscribe<sensor_msgs::Image>(
                imgTopics[cameraId],
                100,
                boost::bind(imgHandlerForCamera, _1, static_cast<int>(cameraId))));
        }
        ros::Subscriber subLidarCloud = n.subscribe<sensor_msgs::PointCloud2>(lidarTopic, 20, lidarCloudHandler);

        started = clock();

        ros::Rate rate(30);
        while(ros::ok())
        {
        	ros::spinOnce();
        	process();

            rate.sleep();
        }

        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->printResult(resultPath);

        int numFramesProcessed = currentId;
        double numSecondsProcessed = fabs(timestamp - firstFrameTime);
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        const double fpsProcessed = numFramesProcessed > 0 && numSecondsProcessed > 0.0 ?
            numFramesProcessed / numSecondsProcessed : 0.0;
        const double msPerFrameSingle = numFramesProcessed > 0 ?
            MilliSecondsTakenSingle / numFramesProcessed : 0.0;
        const double msPerFrameMT = numFramesProcessed > 0 ?
            MilliSecondsTakenMT / static_cast<double>(numFramesProcessed) : 0.0;
        const double realtimeFactorSingle = numSecondsProcessed > 0.0 && MilliSecondsTakenSingle > 0.0 ?
            1000.0 / (MilliSecondsTakenSingle / numSecondsProcessed) : 0.0;
        const double realtimeFactorMT = numSecondsProcessed > 0.0 && MilliSecondsTakenMT > 0.0 ?
            1000.0 / (MilliSecondsTakenMT / numSecondsProcessed) : 0.0;

        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, fpsProcessed,
                msPerFrameSingle,
                msPerFrameMT,
                realtimeFactorSingle,
                realtimeFactorMT);

        if(!resultPath.empty())
        {
            const std::string timingPath = withSuffixAndExtensionLocal(resultPath, "_timing", ".txt");
            std::ofstream timingLog(timingPath.c_str(), std::ios::trunc | std::ios::out);
            if(timingLog.is_open())
            {
                timingLog << "frames " << numFramesProcessed << "\n";
                timingLog << "dataset_duration_s " << numSecondsProcessed << "\n";
                timingLog << "dataset_fps " << fpsProcessed << "\n";
                timingLog << "cpu_time_ms " << MilliSecondsTakenSingle << "\n";
                timingLog << "wall_time_ms " << MilliSecondsTakenMT << "\n";
                timingLog << "ms_per_frame_single_core " << msPerFrameSingle << "\n";
                timingLog << "ms_per_frame_multi_core " << msPerFrameMT << "\n";
                timingLog << "realtime_factor_single_core " << realtimeFactorSingle << "\n";
                timingLog << "realtime_factor_multi_core " << realtimeFactorMT << "\n";
                timingLog.flush();
            }
            else
            {
                printf("warning: could not open timing log at %s\n", timingPath.c_str());
            }
        }

        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*currentId) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)currentId << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("DSO OVER!\n");

	ros::spin();
	return 0;
}
