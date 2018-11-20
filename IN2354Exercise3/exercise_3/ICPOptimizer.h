#pragma once

#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"

/**
 * ICP optimizer.
 */
class ICPOptimizer {
public:
	ICPOptimizer() : 
		m_bUsePointToPlaneConstraints{ false },
		m_nIterations{ 20 },
		m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
	{ }

	void setMatchingMaxDistance(float maxDistance) {
		m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
	}

	void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
		m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
	}

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}

	Matrix4f estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f initialPose = Matrix4f::Identity()) {
		// Build the index of the FLANN tree (for fast nearest neighbor lookup).
		m_nearestNeighborSearch->buildIndex(target.getPoints());

		// The initial estimate can be given as an argument.
		Matrix4f estimatedPose = initialPose;

		for (int i = 0; i < m_nIterations; ++i) {
			// Compute the matches.
			std::cout << "Matching points ..." << std::endl;
			clock_t begin = clock();

			auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
			auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
			pruneCorrespondences(transformedNormals, target.getNormals(), matches);

			clock_t end = clock();
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

			std::vector<Vector3f> sourcePoints;
			std::vector<Vector3f> targetPoints;
			std::vector<Vector3f> targetNormals;

			// TODO: Add all matches to the sourcePoints and targetPoints vector, so that
			//       sourcePoints[i] matches targetPoints[i]. For every source point, the
			//       'matches' vector holds the index of the matching target point and target normal.

			for (int i = 0; i < transformedPoints.size(); i++) {
				if (matches[i].idx != -1) {
					sourcePoints.push_back(transformedPoints[i]);
					targetPoints.push_back(target.getPoints()[matches[i].idx]);
					targetNormals.push_back(target.getNormals()[matches[i].idx]);
				}
			}

      // Estimate the new pose
 			if (m_bUsePointToPlaneConstraints) {
				estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals) * estimatedPose;
			}
			else {
				estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
			}

			std::cout << "Optimization iteration done." << std::endl;
		}

		return estimatedPose;
	}

private:
	bool m_bUsePointToPlaneConstraints;
	unsigned m_nIterations;
	std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

	std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
		std::vector<Vector3f> transformedPoints;
		transformedPoints.reserve(sourcePoints.size());

		const auto rotation = pose.block(0, 0, 3, 3);
		const auto translation = pose.block(0, 3, 3, 1);

		for (const auto& point : sourcePoints) {
			transformedPoints.push_back(rotation * point + translation);
		}

		return transformedPoints;
	}

	std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
		std::vector<Vector3f> transformedNormals;
		transformedNormals.reserve(sourceNormals.size());

		const auto rotation = pose.block(0, 0, 3, 3);

		for (const auto& normal : sourceNormals) {
			transformedNormals.push_back(rotation.inverse().transpose() * normal);
		}

		return transformedNormals;
	}

	void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
		const unsigned nPoints = sourceNormals.size();

		for (unsigned i = 0; i < nPoints; i++) {
			Match& match = matches[i];
			if (match.idx >= 0) {
				const auto& sourceNormal = sourceNormals[i];
				const auto& targetNormal = targetNormals[match.idx];

				// TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
				double cos_val = sourceNormal.dot(targetNormal) / (sourceNormal.norm()*targetNormal.norm());
				if (cos_val < 0.5)  match.idx = -1;
			}
		}
	}

	Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ProcrustesAligner procrustAligner;
		Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

		return estimatedPose;
	}

	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
		const unsigned nPoints = sourcePoints.size();

		// Build the system
		MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
		VectorXf b = VectorXf::Zero(4 * nPoints);

		for (unsigned i = 0; i < nPoints; i++) {
			const auto& s = sourcePoints[i];
			const auto& d = targetPoints[i];
			const auto& n = targetNormals[i];

			// TODO: Add the point-to-plane constraints to the system
      
			A(i * 4, 0) = n.z()*s.y() - n.y()*s.z();
			A(i * 4, 1) = n.x()*s.z() - n.z()*s.x();
			A(i * 4, 2) = n.y()*s.x() - n.x()*s.y();

			A(i * 4, 3) = n.x();
			A(i * 4, 4) = n.y();
			A(i * 4, 5) = n.z();

			b(4 * i, 0) = n.x()*d.x() + n.y()*d.y() + n.z()*d.z() - (n.x()*s.x() + n.y()*s.y() + n.z()*s.z());

			// TODO: Add the point-to-point constraints to the system

			// Constraint for x
			A(i * 4 + 1, 0) = 0;
			A(i * 4 + 1, 1) = s.z();
			A(i * 4 + 1, 2) = -s.y();

			A(i * 4 + 1, 3) = 1;
			A(i * 4 + 1, 4) = 0;
			A(i * 4 + 1, 5) = 0;

			b(4 * i + 1, 0) = d.x() - s.x();

			// Constraint for y
			A(i * 4 + 2, 0) = -s.z();
			A(i * 4 + 2, 1) = 0;
			A(i * 4 + 2, 2) = s.x();

			A(i * 4 + 2, 3) = 0;
			A(i * 4 + 2, 4) = 1;
			A(i * 4 + 2, 5) = 0;

			b(4 * i + 2, 0) = d.y() - s.y();

			// Constraint for z
			A(i * 4 + 3, 0) = s.y();
			A(i * 4 + 3, 1) = -s.x();
			A(i * 4 + 3, 2) = 0;

			A(i * 4 + 3, 3) = 0;
			A(i * 4 + 3, 4) = 0;
			A(i * 4 + 3, 5) = 1;

			b(4 * i + 3, 0) = d.z() - s.z();
		}

		// TODO: Solve the system
		VectorXf x(6);

		JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
		x = svd.solve(b);
		
		float alpha = x(0), beta = x(1), gamma = x(2);

		// Build the pose matrix
		Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
			                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
			                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

		Vector3f translation = x.tail(3);

		// TODO: Build the pose matrix using the rotation and translation matrices
		Matrix4f estimatedPose = Matrix4f::Identity();
		
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = translation;

		return estimatedPose;
	}
};
