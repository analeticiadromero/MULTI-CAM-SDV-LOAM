#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace sdv_loam
{

void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	linearizeAll_Reductor(visualState, fixLinearization, toRemove, min, max, stats, tid);
}

void FullSystem::linearizeAll_Reductor(VisualState& visualState, bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		PointFrameResidual* r = visualState.activeResiduals[k];
		(*stats)[0] += r->linearize(&Hcalib);

		if(fixLinearization)
		{
			r->applyRes(true);

			if(r->efResidual->isActive())
			{
				if(r->isNew)
				{
					PointHessian* p = r->point;
					Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
					Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth.
					float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();		// 0.01 = one pixel.
					
					if(relBS > p->maxRelBaseline)
						p->maxRelBaseline = relBS;

					p->numGoodResiduals++;
				}
			}
			else
			{
				toRemove[tid].push_back(visualState.activeResiduals[k]);
			}
		}
	}
}

void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	applyRes_Reductor(visualState, copyJacobians, min, max, stats, tid);
}

void FullSystem::applyRes_Reductor(VisualState& visualState, bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
		visualState.activeResiduals[k]->applyRes(true);
}

void FullSystem::setNewFrameEnergyTH()
{
	setNewFrameEnergyTH(visualState);
}

void FullSystem::setNewFrameEnergyTH(VisualState& visualState)
{

	// collect all residuals and make decision on TH.
	visualState.allResVec.clear();
	visualState.allResVec.reserve(visualState.activeResiduals.size()*2);
	FrameHessian* newFrame = visualState.frameHessians.back();

	for(PointFrameResidual* r : visualState.activeResiduals)
		if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
		{
			visualState.allResVec.push_back(r->state_NewEnergyWithOutlier);

		}

	if(visualState.allResVec.size()==0)
	{
		newFrame->frameEnergyTH = 12*12*patternNum;
		return;		// should never happen, but lets make sure.
	}


	int nthIdx = setting_frameEnergyTHN*visualState.allResVec.size();

	assert(nthIdx < (int)visualState.allResVec.size());
	assert(setting_frameEnergyTHN < 1);

	std::nth_element(visualState.allResVec.begin(), visualState.allResVec.begin()+nthIdx, visualState.allResVec.end());
	float nthElement = sqrtf(visualState.allResVec[nthIdx]);

    newFrame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
	newFrame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
	newFrame->frameEnergyTH = newFrame->frameEnergyTH*newFrame->frameEnergyTH;
	newFrame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;
}

Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
	return linearizeAll(visualState, fixLinearization);
}

Vec3 FullSystem::linearizeAll(VisualState& visualState, bool fixLinearization)
{
	double lastEnergyP = 0;
	double lastEnergyR = 0;
	double num = 0;


	std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

	if(multiThreading)
	{
		treadReduce.reduce(boost::bind(
			static_cast<void (FullSystem::*)(VisualState&, bool, std::vector<PointFrameResidual*>*, int, int, Vec10*, int)>(&FullSystem::linearizeAll_Reductor),
			this,
			boost::ref(visualState),
			fixLinearization,
			toRemove,
			_1, _2, _3, _4), 0, visualState.activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else
	{
		Vec10 stats;
		linearizeAll_Reductor(visualState, fixLinearization, toRemove, 0,visualState.activeResiduals.size(),&stats,0);
		lastEnergyP = stats[0];
	}


	setNewFrameEnergyTH(visualState);


	if(fixLinearization)
	{
		for(PointFrameResidual* r : visualState.activeResiduals)
		{
			PointHessian* ph = r->point;
			if(ph->lastResiduals[0].first == r)
				ph->lastResiduals[0].second = r->state_state;
			else if(ph->lastResiduals[1].first == r)
				ph->lastResiduals[1].second = r->state_state;
		}
		
		for(int i=0;i<NUM_THREADS;i++)
		{
			for(PointFrameResidual* r : toRemove[i])
			{
				PointHessian* ph = r->point;

				if(ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].first=0;
				else if(ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].first=0;

				for(unsigned int k=0; k<ph->residuals.size();k++)
						if(ph->residuals[k] == r)
						{
							visualState.ef->dropResidual(r->efResidual);
							deleteOut<PointFrameResidual>(ph->residuals,k);
							break;
						}
			}
		}
	}

	return Vec3(lastEnergyP, lastEnergyR, num);
}




// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
	return doStepFromBackup(visualState, stepfacC, stepfacT, stepfacR, stepfacA, stepfacD);
}

bool FullSystem::doStepFromBackup(VisualState& visualState, float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);
	pstepfac.segment<4>(6).setConstant(stepfacA);


	float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

	float sumNID=0;

	Eigen::Matrix<double, 4, 1> tmp;
	tmp.setZero();

	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
		for(FrameHessian* fh : visualState.frameHessians)
		{
			(fh->step).segment<4>(6) = tmp;

			Vec10 step = fh->step;
			step.head<6>() += 0.5f*(fh->step_backup.head<6>());

			fh->setState(fh->state_backup + step);

			sumT += step.segment<3>(0).squaredNorm();
			sumR += step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				float step = ph->step+0.5f*(ph->step_backup);
				if(!std::isfinite(step)) step = 0;
				const float newIdepth = ph->idepth_backup + step;
				if(!std::isfinite(newIdepth)) step = 0;
				ph->setIdepth(ph->idepth_backup + step);
				sumID += step*step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + step); 
			}
		}
	}
	else
	{
		Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);

		for(FrameHessian* fh : visualState.frameHessians)
		{
			(fh->step).segment<4>(6) = tmp;

			fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));

			sumT += fh->step.segment<3>(0).squaredNorm();
			sumR += fh->step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				float step = stepfacD*ph->step;
				if(!std::isfinite(step)) step = 0;
				const float newIdepth = ph->idepth_backup + step;
				if(!std::isfinite(newIdepth)) step = 0;
				ph->setIdepth(ph->idepth_backup + step);
				sumID += step*step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + step);
			}
		}
	}

	sumR /= visualState.frameHessians.size();
	sumT /= visualState.frameHessians.size();
	sumID /= numID;
	sumNID /= numID;

    if(!setting_debugout_runquiet)
        printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
                sqrtf(sumA) / (0.0005*setting_thOptIterations),
                sqrtf(sumB) / (0.00005*setting_thOptIterations),
                sqrtf(sumR) / (0.00005*setting_thOptIterations),
                sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));


	EFDeltaValid=false;
	setPrecalcValues(visualState);

	return 	sqrtf(sumR) < 0.00005*setting_thOptIterations &&
			sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
}



// sets linearization point.
void FullSystem::backupState(bool backupLastStep)
{
	backupState(visualState, backupLastStep);
}

void FullSystem::backupState(VisualState& visualState, bool backupLastStep)
{
	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		if(backupLastStep)
		{
			Hcalib.step_backup = Hcalib.step;
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : visualState.frameHessians)
			{
				fh->step_backup = fh->step;
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup = ph->step;
				}
			}
		}
		else
		{
			Hcalib.step_backup.setZero();
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : visualState.frameHessians)
			{
				fh->step_backup.setZero();
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup=0;
				}
			}
		}
	}
	else
	{
		Hcalib.value_backup = Hcalib.value;
		for(FrameHessian* fh : visualState.frameHessians)
		{
			fh->state_backup = fh->get_state();
			for(PointHessian* ph : fh->pointHessians)
				ph->idepth_backup = ph->idepth;
		}
	}
}

// sets linearization point.
void FullSystem::loadSateBackup()
{
	loadSateBackup(visualState);
}

void FullSystem::loadSateBackup(VisualState& visualState)
{
	Hcalib.setValue(Hcalib.value_backup);
	for(FrameHessian* fh : visualState.frameHessians)
	{
		fh->setState(fh->state_backup);
		for(PointHessian* ph : fh->pointHessians)
		{
			ph->setIdepth(ph->idepth_backup);

            ph->setIdepthZero(ph->idepth_backup);
		}

	}


	EFDeltaValid=false;
	setPrecalcValues(visualState);
}

double FullSystem::calcMEnergy()
{
	return calcMEnergy(visualState);
}

double FullSystem::calcMEnergy(VisualState& visualState)
{
	if(setting_forceAceptStep) return 0;

	return visualState.ef->calcMEnergyF();
}


void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printOptRes(visualState, res, resL, resM, resPrior, LExact, a, b);
}

void FullSystem::printOptRes(VisualState& visualState, const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0],
			sqrtf((float)(res[0] / (patternNum*visualState.ef->resInA))),
			visualState.ef->resInA,
			visualState.ef->resInM,
			a,
			b
	);

}

float FullSystem::optimize(int mnumOptIts)
{
	return optimize(visualState, mnumOptIts);
}

float FullSystem::optimize(VisualState& visualState, int mnumOptIts)
{

	if(visualState.frameHessians.size() < 2) return 0;
	if(visualState.frameHessians.size() < 3) mnumOptIts = 100;
	if(visualState.frameHessians.size() < 4) mnumOptIts = 75;

	visualState.activeResiduals.clear();
	int numPoints = 0;
	int numLRes = 0;
	for(FrameHessian* fh : visualState.frameHessians)
		for(PointHessian* ph : fh->pointHessians)
		{
			for(PointFrameResidual* r : ph->residuals)
			{
				if(!r->efResidual->isLinearized)
				{
					visualState.activeResiduals.push_back(r);
					r->resetOOB();
				}
				else
					numLRes++;
			}
			numPoints++;
		}

    if(!setting_debugout_runquiet)
        printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",visualState.ef->nPoints,(int)visualState.activeResiduals.size(), numLRes);

	Vec3 lastEnergy = linearizeAll(visualState, false);  

	double lastEnergyL = calcLEnergy(visualState);
	double lastEnergyM = calcMEnergy(visualState);

	if(multiThreading)
		treadReduce.reduce(boost::bind(
			static_cast<void (FullSystem::*)(VisualState&, bool, int, int, Vec10*, int)>(&FullSystem::applyRes_Reductor),
			this,
			boost::ref(visualState),
			true,
			_1, _2, _3, _4), 0, visualState.activeResiduals.size(), 50);
	else
		applyRes_Reductor(visualState, true,0,visualState.activeResiduals.size(),0,0);

    if(!setting_debugout_runquiet)
    {
        printf("Initial Error       \t");
        printOptRes(visualState, lastEnergy, lastEnergyL, lastEnergyM, 0, 0, visualState.frameHessians.back()->aff_g2l().a, visualState.frameHessians.back()->aff_g2l().b);
    }

	debugPlotTracking();

	double lambda = 1e-1;
	float stepsize=1;

	VecX previousX = VecX::Constant(CPARS+ 6*visualState.frameHessians.size(), NAN);
	for(int iteration=0;iteration<mnumOptIts;iteration++)
	{
		// solve!
		backupState(visualState, iteration!=0);

		solveSystem(visualState, iteration, lambda);
		double incDirChange = (1e-20 + previousX.dot(visualState.ef->lastX)) / (1e-20 + previousX.norm() * visualState.ef->lastX.norm());
		previousX = visualState.ef->lastX;

		if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) 
		{
			float newStepsize = exp(incDirChange*1.4);
			if(incDirChange<0 && stepsize>1) stepsize=1;

			stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
			if(stepsize > 2) stepsize=2;
			if(stepsize <0.25) stepsize=0.25;
		}

		bool canbreak = doStepFromBackup(visualState, stepsize,stepsize,stepsize,stepsize,stepsize);

		// eval new energy!
		Vec3 newEnergy = linearizeAll(visualState, false);
		double newEnergyL = calcLEnergy(visualState);
		double newEnergyM = calcMEnergy(visualState);

        if(!setting_debugout_runquiet)
        {
            printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
				(newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
						lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				incDirChange,
				stepsize);
            printOptRes(visualState, newEnergy, newEnergyL, newEnergyM , 0, 0, visualState.frameHessians.back()->aff_g2l().a, visualState.frameHessians.back()->aff_g2l().b);
        }

		if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
		{  
			if(multiThreading)
				treadReduce.reduce(boost::bind(
					static_cast<void (FullSystem::*)(VisualState&, bool, int, int, Vec10*, int)>(&FullSystem::applyRes_Reductor),
					this,
					boost::ref(visualState),
					true,
					_1, _2, _3, _4), 0, visualState.activeResiduals.size(), 50);
			else
				applyRes_Reductor(visualState, true,0,visualState.activeResiduals.size(),0,0);

			lastEnergy = newEnergy;
			lastEnergyL = newEnergyL;
			lastEnergyM = newEnergyM;

			lambda *= 0.25;
		}
		else
		{
			loadSateBackup(visualState);
			lastEnergy = linearizeAll(visualState, false);
			lastEnergyL = calcLEnergy(visualState);
			lastEnergyM = calcMEnergy(visualState);
			lambda *= 1e2;
		}


		if(canbreak && iteration >= setting_minOptIterations) break;
	}

	Vec10 newStateZero = Vec10::Zero();
	newStateZero.segment<2>(6) = visualState.frameHessians.back()->get_state().segment<2>(6); 

	visualState.frameHessians.back()->setEvalPT(visualState.frameHessians.back()->PRE_worldToCam,
			newStateZero);
	EFDeltaValid=false;
	EFAdjointsValid=false;
	visualState.ef->setAdjointsF(&Hcalib);
	setPrecalcValues(visualState);

	lastEnergy = linearizeAll(visualState, true);

	if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
    {
        printf("KF Tracking failed: LOST!\n");
		visualState.isLost=true;
    }

	statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / visualState.ef->resInA));

	if(calibLog != 0)
	{	
		(*calibLog) << Hcalib.value_scaled.transpose() <<
				" " << visualState.frameHessians.back()->get_state_scaled().transpose() <<
				" " << sqrtf((float)(lastEnergy[0] / visualState.ef->resInA)) <<
				" " << visualState.ef->resInM << "\n";

		calibLog->flush();
	}

		{
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			for(FrameHessian* fh : visualState.frameHessians)
			{
				// The optimized visual state is still expressed as T_WC in the legacy pipeline.
				fh->shell->setT_WC(fh->PRE_camToWorld);
				fh->shell->aff_g2l = fh->aff_g2l();
			}
		}

	syncRigStateFromFrameShell(visualState.frameHessians.back()->shell);

	debugPlotTracking();

	return sqrtf((float)(lastEnergy[0] / (visualState.ef->resInA)));
}

void FullSystem::solveSystem(int iteration, double lambda)
{
	solveSystem(visualState, iteration, lambda);
}

void FullSystem::solveSystem(VisualState& visualState, int iteration, double lambda)
{
	visualState.ef->lastNullspaces_forLogging = getNullspaces(
			visualState,
			visualState.ef->lastNullspaces_pose,
			visualState.ef->lastNullspaces_scale,
			visualState.ef->lastNullspaces_affA,
			visualState.ef->lastNullspaces_affB);

	visualState.ef->solveSystemF(iteration, lambda,&Hcalib);
}

double FullSystem::calcLEnergy()
{
	return calcLEnergy(visualState);
}

double FullSystem::calcLEnergy(VisualState& visualState)
{
	if(setting_forceAceptStep) return 0;

	double Ef = visualState.ef->calcLEnergyF_MT();
	return Ef;

}

void FullSystem::removeOutliers()
{
	removeOutliers(visualState);
}

void FullSystem::removeOutliers(VisualState& visualState)
{
	int numPointsDropped=0;
	for(FrameHessian* fh : visualState.frameHessians)
	{
		for(unsigned int i=0;i<fh->pointHessians.size();i++)
		{
			PointHessian* ph = fh->pointHessians[i];
			if(ph==0) continue;

			if(ph->residuals.size() == 0)
			{
				fh->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				fh->pointHessians[i] = fh->pointHessians.back();
				fh->pointHessians.pop_back();
				i--;
				numPointsDropped++;
			}
		}
	}
	visualState.ef->dropPointsF();
}

std::vector<VecX> FullSystem::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	return getNullspaces(visualState, nullspaces_pose, nullspaces_scale, nullspaces_affA, nullspaces_affB);
}

std::vector<VecX> FullSystem::getNullspaces(
		VisualState& visualState,
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	nullspaces_pose.clear();  // size: 6; vec: 4+8*n
	nullspaces_scale.clear(); // size: 1; 
	nullspaces_affA.clear();  // size: 1
	nullspaces_affB.clear();  // size: 1 

	int n = CPARS + visualState.frameHessians.size() * 6;
	std::vector<VecX> nullspaces_x0_pre;
	
	for(int i=0;i<6;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : visualState.frameHessians)
		{
			nullspace_x0.segment<6>(CPARS+fh->idx*6) = fh->nullspaces_pose.col(i);
			nullspace_x0.segment<3>(CPARS+fh->idx*6) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS+fh->idx*6+3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_pose.push_back(nullspace_x0);
	}

	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	for(FrameHessian* fh : visualState.frameHessians)
	{
		nullspace_x0.segment<6>(CPARS+fh->idx*6) = fh->nullspaces_scale;
		nullspace_x0.segment<3>(CPARS+fh->idx*6) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS+fh->idx*6+3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
