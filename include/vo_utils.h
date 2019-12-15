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

#include <set>

#include "common_types.h"

#include "calibration.h"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.

  for (auto& landmark : landmarks) {
      TrackId trackId = landmark.first;
      Eigen::Vector3d landmark_position_in_world_coordinate = landmark.second.p; // In world coordinate
      Eigen::Vector3d landmark_position_in_camera_coordinate = current_pose.inverse() * landmark_position_in_world_coordinate;

      if (landmark_position_in_camera_coordinate.z() >= cam_z_threshold) {
          Eigen::Vector2d landmark_projection_in_2d_image_plane = cam->project(landmark_position_in_camera_coordinate);

          if (
                  (landmark_projection_in_2d_image_plane.x() >= 0) &&
                  (landmark_projection_in_2d_image_plane.x() < 752) &&
                  (landmark_projection_in_2d_image_plane.y() >= 0) &&
                  (landmark_projection_in_2d_image_plane.y() < 480)) {

              projected_points.push_back(landmark_projection_in_2d_image_plane);
              projected_track_ids.push_back(trackId);
          }
      }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_max_dist and feature_match_test_next_best
  // should be used to filter outliers the same way as in exercise 3.

  for (int i = 0; i < kdl.corners.size(); i++) {
      int first_best = 1000;
      int second_best = 1000;
      int save_index = -1;

      std::bitset<256> corner_descriptor_1 = kdl.corner_descriptors.at(i);

      for (int j = 0; j < projected_points.size(); j++) {
        double distance = std::sqrt(
                    ((kdl.corners.at(i).x() - projected_points.at(j).x()) *
                     (kdl.corners.at(i).x() - projected_points.at(j).x())) +
                    ((kdl.corners.at(i).y() - projected_points.at(j).y()) *
                     (kdl.corners.at(i).y() - projected_points.at(j).y())));

        if (distance < match_max_dist_2d) {
            TrackId trackId = projected_track_ids.at(j);
            Landmark landmark = landmarks.at(trackId);

            int minHamdist = 1000;
            for (auto observation : landmark.obs) {
                TimeCamId tcid = observation.first;
                FeatureId featureId = observation.second;

                KeypointsData keypointData = feature_corners.at(tcid);
                std::bitset<256> corner_descriptor_2 = keypointData.corner_descriptors.at(featureId);


                std::bitset<256> diff_bitset = (corner_descriptor_1 ^ corner_descriptor_2);
                int hamDist = diff_bitset.count();

                if (hamDist < minHamdist) {
                    minHamdist = hamDist;
                }

            }

            if (minHamdist < first_best)
            {
                second_best = first_best;
                first_best = minHamdist;
                save_index = projected_track_ids[j];
            }
            else if (minHamdist < second_best)
            {
                second_best = minHamdist;
            }
        }
    }

      if ((first_best < feature_match_max_dist) && (second_best > (first_best * feature_match_test_next_best))) {
          md.matches.push_back(std::make_pair(i, save_index));
      }
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  // TODO SHEET 5: Find the pose (T_w_c) and the inliers using the landmark to
  // keypoints matches and PnP. This should be similar to the localize_camera in
  // exercise 4 but in this execise we don't explicitelly have tracks.

  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (auto& match : md.matches) {

      FeatureId featureId = match.first;

      Eigen::Vector2d p_2d = kdl.corners.at(featureId);

      bearingVectors.push_back(cam->unproject(p_2d).normalized());

      opengv::point_t point = landmarks.at(match.second).p;
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

  std::cout << "ransac.inliers_size: " << ransac.inliers_.size() << "\n";

  inliers = ransac.inliers_;
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains landmark to map
  // matches for the left camera (camera 0). The inliers vector contains all
  // inliers in md that were used to compute the pose T_w_c0. For all inliers
  // add the observations to the existing ladmarks (if the left point is in
  // md_stereo.inliers then add both observations). For all stereo observations
  // that were not added to the existing landmarks triangulate and add new
  // landmarks. Here next_landmark_id is a running index of the landmarks, so
  // after adding a new landmark you should always increase next_landmark_id
  // by 1.

  std::map<FeatureId, FeatureId> myStereo;

  for (auto& inlier : md_stereo.inliers) {
      myStereo[inlier.first] = inlier.second;
  }

  for (int i = 0; i < inliers.size(); i++) {
      FeatureId featureId0 = md.matches.at(inliers.at(i)).first;
      FeatureId featureId1 = md.matches.at(inliers.at(i)).second;

      Landmark& landmark = landmarks[featureId1];
      landmark.obs[tcidl] = featureId0;

      auto found = myStereo.find(featureId0);
      if (found != myStereo.end()) {
          landmark.obs[tcidr] = found->second;
      }
      myStereo.erase(featureId0);
  }

  std::vector<std::pair<FeatureId, FeatureId>> my_inliers;

  opengv::bearingVectors_t bearingVectors0;
  opengv::bearingVectors_t bearingVectors1;

  for (auto& matchData : myStereo) {

      FeatureId featureId0 = matchData.first;
      FeatureId featureId1 = matchData.second;

      my_inliers.push_back(std::make_pair(featureId0, featureId1));

      Eigen::Vector2d p0_2d = kdl.corners[featureId0];
      Eigen::Vector2d p1_2d = kdl.corners[featureId1];

      bearingVectors0.push_back(calib_cam.intrinsics[tcidl.second]->unproject(p0_2d).normalized());
      bearingVectors1.push_back(calib_cam.intrinsics[tcidr.second]->unproject(p1_2d).normalized());

  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearingVectors0, bearingVectors1, t_0_1, R_0_1);

  for (int index = 0; index < bearingVectors0.size(); index++) {

    opengv::point_t point = opengv::triangulation::triangulate(adapter, index);

    Landmark landmark;
    landmark.p = T_w_c0 * point;
    landmark.obs[tcidl] = my_inliers.at(index).first;
    landmark.obs[tcidr] = my_inliers.at(index).second;

    landmarks[next_landmark_id] = landmark;
    next_landmark_id++;
  }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  while (true) {
      if (kf_frames.size() > max_num_kfs) {

          FrameId frameId = *(kf_frames.begin());

          cameras.erase(TimeCamId(frameId, tcidl.second));
          cameras.erase(TimeCamId(frameId, (tcidl.second + 1)));

          for (auto& landmark : landmarks) {
            landmark.second.obs.erase(TimeCamId(frameId, tcidl.second));
            landmark.second.obs.erase(TimeCamId(frameId, (tcidl.second + 1)));

          }

          kf_frames.erase(frameId);
      }
      else {
          break;
      }
    }

}
