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

#include <sophus/se3.hpp>
#include <Eigen/Dense>

#include "common_types.h"

using namespace Eigen;
using namespace std;

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {

    Eigen::Matrix<T, 3, 3> res;

    T theta_square = (xi.transpose() * xi);
    T theta = sqrt(theta_square);

    if (theta != T(0)) {

        T cos_theta = cos(theta);
        T sin_theta = sin(theta);

        Vector3d a = xi / theta;

        // Making Skew Symmetric Matrix
        Matrix<T, 3, 3> a_skew;
        a_skew << 0, -a(2), a(1),
                a(2), 0, -a(0),
                -a(1), a(0), 0;

        // Calculating Result
        res = (cos_theta * Matrix<T, 3, 3>::Identity()) + ((1 - cos_theta) * (a * a.transpose())) + (sin_theta * a_skew) ;
    }
    else
    {
        res = Matrix<T, 3, 3>::Identity();
    }

  return res;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {

    Eigen::Matrix<T, 3, 1> res;

    T theta = acos(( mat.trace() - T(1)) / T(2));

    if(theta != T(0))
    {
        Matrix<T, 3, 1> axis;
        axis << (mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)), (mat(1, 0) - mat(0, 1));

        Matrix<T, 3, 1> a;
        a = ((T(1) / (T(2) * sin(theta))) * axis);
        res = theta * a;
    }
    else
    {
       res.setZero();
    }


  return res;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {

    Eigen::Matrix<T, 4, 4> res;

    Matrix<T, 3, 1> phi, roh;
    roh << xi(0), xi(1), xi(2);
    phi << xi(3), xi(4), xi(5);

    Matrix<T, 3, 3> R = user_implemented_expmap(phi);

    T theta_square = (phi.transpose() * phi);
    T theta = sqrt(theta_square);

    if (theta != T(0))
    {
        Vector3d a = phi / theta;

        // Making Skew Symmetric Matrix
        Matrix<T, 3, 3> a_skew;
        a_skew << 0, -a(2), a(1),
                a(2), 0, -a(0),
                -a(1), a(0), 0;

        Matrix<T, 3, 3> J =
                ((sin(theta) / theta) * Matrix<T, 3, 3>::Identity()) +
                ((T(1) - (sin(theta) / theta)) * (a * a.transpose())) +
                (((T(1) - cos(theta)) / theta) * a_skew);

        res.topLeftCorner(3, 3) = R;
        res.topRightCorner(3, 1) = (J * roh);
        res.bottomLeftCorner(1, 3) = Matrix<T, 1, 3>::Zero();
        res.bottomRightCorner(1, 1) << T(1);
    }
    else
    {
        res = Matrix<T, 4, 4>::Identity();
    }


  return res;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {

  Eigen::Matrix<T, 6, 1> res;
  Eigen::Matrix<T, 3, 1> rot;
  Eigen::Matrix<T, 3, 1> roh;

  Eigen::Matrix<T, 3, 3> mat_sub = mat.topLeftCorner(3, 3);

  T theta = acos(( mat_sub.trace() - T(1)) / T(2));

  if(theta != T(0))
  {
      Matrix<T, 3, 1> axis;
      axis << (mat_sub(2, 1) - mat_sub(1, 2)),
              (mat_sub(0, 2) - mat_sub(2, 0)),
              (mat_sub(1, 0) - mat_sub(0, 1));

      Matrix<T, 3, 1> a;
      a = ((T(1) / (T(2) * sin(theta))) * axis);
      rot = theta * a;

      // Calculating roh
      Matrix<T, 3, 1> t = mat.topRightCorner(3, 1);

      // Making Skew Symmetric Matrix
      Matrix<T, 3, 3> a_skew;
      a_skew << 0, -a(2), a(1),
              a(2), 0, -a(0),
              -a(1), a(0), 0;

      Matrix<T, 3, 3> J =
              ((sin(theta) / theta) * Matrix<T, 3, 3>::Identity()) +
              ((T(1) - (sin(theta) / theta)) * (a * a.transpose())) +
              (((T(1) - cos(theta)) / theta) * a_skew);

      roh = (J.inverse() * t);
  }
  else
  {
     rot.setZero();
     roh.setZero();
  }

  res << roh, rot;

  return res;
}
