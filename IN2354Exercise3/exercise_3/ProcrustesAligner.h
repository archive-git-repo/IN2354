#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");

		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);

		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean);

		// To apply the pose to point x on shape X in the case of Procrustes, we execute:
		// 1. Translation of a point to the shape Y: x' = x + t
		// 2. Rotation of the point around the mean of shape Y: 
		//    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t - R yMean + yMean)

		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;

		return estimatedPose;
	}

private:
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		// TODO: Compute the mean of input points.

		Vector3f mean = Vector3f::Random();
		float sum1, sum2, sum3;
		for (auto i = 0; i < points.size(); i++)
		{
			sum1 = sum1 + points[i][0];
			sum2 = sum2 + points[i][1];
			sum3 = sum3 + points[i][2];
		}
		mean[0] = sum1 / points.size();
		mean[2] = sum2 / points.size();
		mean[2] = sum3 / points.size();
		return mean;
	}

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		// TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Important: The covariance matrices should contain mean-centered source/target points.

		MatrixXd sourcePM(sourcePoints.size()+1, 3);
		MatrixXd targetPM(targetPoints.size()+1, 3);
		for (auto i = 0; i < sourcePoints.size(); i++)
		{
			sourcePM << sourcePoints[i];
			targetPM << targetPoints[i];
		}
		sourcePM << sourceMean;
		targetPM << targetMean;

		JacobiSVD<Matrix3f> svd(targetPM.transpose()*sourcePM, ComputeThinU | ComputeThinV);
		auto rotation = svd.matrixU()*(svd.matrixV().transpose());
		return rotation;
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
		// TODO: Compute the translation vector from source to target points.

		Vector3f translation = Vector3f::Random();
		for (auto i = 0; i < targetMean.size(); i++)
		{
			translation[i]=targetMean[i] - sourceMean[i];
		}
		return translation;
	}
};