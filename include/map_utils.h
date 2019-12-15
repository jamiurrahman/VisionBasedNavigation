/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include "common_types.h"
#include "serialization.h"

#include "local_parameterization_se3.hpp"
#include "reprojection.h"

#include "tracks.h"

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const TimeCamId& tcid0,
                                   const TimeCamId& tcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<TimeCamId> tcids = {tcid0, tcid1};
  if (!GetTracksInImages(tcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map
  // Start
  opengv::bearingVectors_t bearingVectors0;
  opengv::bearingVectors_t bearingVectors1;

  for (TrackId shared_track_id : shared_track_ids) {

    if (landmarks.find(shared_track_id) == landmarks.end()) {

      new_track_ids.push_back(shared_track_id);

      FeatureId featureId0 = feature_tracks.at(shared_track_id).at(tcid0);
      FeatureId featureId1 = feature_tracks.at(shared_track_id).at(tcid1);

      Eigen::Vector2d p0_2d = feature_corners.at(tcid0).corners[featureId0];
      Eigen::Vector2d p1_2d = feature_corners.at(tcid1).corners[featureId1];

      bearingVectors0.push_back(calib_cam.intrinsics[tcid0.second]->unproject(p0_2d).normalized());
      bearingVectors1.push_back(calib_cam.intrinsics[tcid1.second]->unproject(p1_2d).normalized());

    }
  }

  Sophus::SE3d relative_pose;
  relative_pose = cameras.at(tcid0).T_w_c.inverse() * cameras.at(tcid1).T_w_c;

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors0, bearingVectors1, relative_pose.translation(), relative_pose.rotationMatrix());

  for (int index = 0; index < new_track_ids.size(); index++) {

    opengv::point_t point = opengv::triangulation::triangulate(adapter, index);

    Landmark landmark;
    landmark.p = cameras.at(tcid0).T_w_c * point;

    for (auto iterator = feature_tracks.at(new_track_ids[index]).begin();
         iterator != feature_tracks.at(new_track_ids[index]).end(); iterator++) {

      auto find = cameras.find(iterator->first);
      if (find != cameras.end()) {
        landmark.obs[iterator->first] = iterator->second;
      }
    }
    landmarks[new_track_ids[index]] = landmark;
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const TimeCamId& tcid0,
                                       const TimeCamId& tcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(tcid0.first == tcid1.first && tcid0.second != tcid1.second)) {
    std::cerr << "Images " << tcid0 << " and " << tcid1
              << " don't for a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)

  std::cout << "Into initialize_scene_from_stereo_pair.\n";
    //cout << "Printing all the values: \n" << tcid0.second << " " << tcid1.second << "\n";


    //Sophus::SE3d test = (Sophus::SE3d)Eigen::Matrix4Xd::Identity();
      cameras[tcid0].T_w_c = (Sophus::SE3d)Eigen::Matrix<double, 4, 4>::Identity();
      cameras[tcid1].T_w_c = calib_cam.T_i_c[tcid1.second];

      add_new_landmarks_between_cams(tcid0, tcid1, calib_cam, feature_corners, feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const TimeCamId& tcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (TrackId shared_track_id : shared_track_ids) {

      FeatureId featureId = feature_tracks.at(shared_track_id).at(tcid);

      Eigen::Vector2d p_2d = feature_corners.at(tcid).corners[featureId];

      bearingVectors.push_back(calib_cam.intrinsics[tcid.second]->unproject(p_2d).normalized());

      opengv::point_t point = landmarks.at(shared_track_id).p;
      points.push_back(point);
  }

  // create the central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(
      bearingVectors, points );
  // create a Ransac object
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
      adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP ) );
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;

  double threshold = 1.0 - cos((reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.threshold_ = threshold;

  ransac.max_iterations_ = 50;
  ransac.computeModel();

    adapter.sett(T_w_c.translation());
    adapter.setR(T_w_c.rotationMatrix());

    const opengv::transformation_t nonlinear_transformation =
        opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

    T_w_c.setRotationMatrix(nonlinear_transformation.block<3, 3>(0, 0));
    T_w_c.translation() = nonlinear_transformation.block<3, 1>(0, 3);

    ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                            ransac.threshold_, ransac.inliers_);

//  std::cout << "ransac.inliers_size: " << ransac.inliers_.size() << "\n";

  for (int j = 0; j < ransac.inliers_.size() ; j++) {
      inlier_track_ids.push_back(shared_track_ids.at(ransac.inliers_[j]));
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<TimeCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;


  // TODO SHEET 4: Setup optimization problem

  for(auto& camera : cameras) {
      problem.AddParameterBlock(
                  camera.second.T_w_c.data(),
                  Sophus::SE3d::num_parameters,
                  new Sophus::test::LocalParameterizationSE3
                  );

      if (fixed_cameras.find(camera.first) != fixed_cameras.end()) {
          problem.SetParameterBlockConstant(cameras.at(camera.first).T_w_c.data());
      }
  }

  for (auto& landmark : landmarks) {
       problem.AddParameterBlock(
                  landmark.second.p.data(),
                  3
                  );

  }

  for (auto& intrinsic : calib_cam.intrinsics) {
          problem.AddParameterBlock(
                  intrinsic->data(),
                  8
                  );

          if(!options.optimize_intrinsics) {
              problem.SetParameterBlockConstant(intrinsic->data());
          }
  }

  for (auto& landmark : landmarks) {

      for (auto& featureTracks : landmark.second.obs) {

          TimeCamId tcid = featureTracks.first;

          ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction
                                         <
                                         BundleAdjustmentReprojectionCostFunctor,
                                         2,
                                         Sophus::SE3d::num_parameters,
                                         3,
                                         8>
                                         (new BundleAdjustmentReprojectionCostFunctor(
                                              feature_corners.at(tcid).corners.at(featureTracks.second),
                                         calib_cam.intrinsics[tcid.second]->name()));

          if (options.use_huber) {
              problem.AddResidualBlock(cost_function, new ceres::HuberLoss(options.huber_parameter),
                                       cameras.at(tcid).T_w_c.data(),
                                       landmark.second.p.data(),
                                       calib_cam.intrinsics[tcid.second]->data());
          }
          else {
              problem.AddResidualBlock(cost_function, NULL,
                                       cameras.at(tcid).T_w_c.data(),
                                       landmark.second.p.data(),
                                       calib_cam.intrinsics[tcid.second]->data());
          }

      }
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}
