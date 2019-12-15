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

#include <memory>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include "common_types.h"

template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() { param.setZero(); }

  PinholeCamera(const VecN& p) { param = p; }

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    //Implementing Pinhole Projection
    if(z != Scalar(0))
    {
        res[0] = ((fx * (x / z)) + cx);
        res[1] = ((fy * (y / z)) + cy);
    }
    else
    {
        res.setZero();
    }

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    Vec3 res;

    //Implementing Pinhole Unprojection
    Scalar mx = Scalar(0);
    Scalar my = Scalar(0);

    if(fx != Scalar(0))
    {
        mx = ((p[0] - cx) / fx);
    }
    if(fy != Scalar(0))
    {
        my = ((p[1] - cy) / fy);
    }

    Scalar denominator = sqrt((mx * mx) + (my * my) + Scalar(1));

    if (denominator != Scalar(0))
    {
        res[0] = (mx / denominator);
        res[1] = (my / denominator);
        res[2] = (Scalar(1) / denominator);
    }
    else
    {
        res.setZero();
    }

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() { param.setZero(); }

  ExtendedUnifiedCamera(const VecN& p) { param = p; }

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;
    \
    //Implementing EUCM Projection
    Scalar d = sqrt((beta * ((x * x) + (y * y))) + (z * z));

    Scalar denominator = ((alpha * d) + ((Scalar(1) - alpha) * z));

    if(denominator != 0.0)
    {
        res[0] = (((fx * x) / denominator) + cx);
        res[1] = (((fy * y) / denominator) + cy);
    }
    else
    {
        res.setZero();
    }

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    Vec3 res;

    //Implementing EUCM Unprojection
    Scalar mx = Scalar(0);
    Scalar my = Scalar(0);
    Scalar mz = Scalar(0);

    if(fx != Scalar(0))
    {
        mx = ((p[0] - cx) / fx);
    }
    if(fy != Scalar(0))
    {
        my = ((p[1] - cy) / fy);
    }

    Scalar r_squre = ((mx * mx) + (my * my));

    Scalar denominator1 = ((alpha * sqrt((Scalar(1) - (((Scalar(2) * alpha) - Scalar(1)) * beta * r_squre)))) + (Scalar(1) - alpha));

    if(denominator1 != Scalar(0))
    {
        mz = ((Scalar(1) - (beta * alpha * alpha * r_squre)) / denominator1);
    }

    Scalar denominator2 = sqrt(r_squre + (mz * mz));

    if (denominator2 != Scalar(0))
    {
        res[0] = (mx / denominator2);
        res[1] = (my / denominator2);
        res[2] = (mz / denominator2);
    }
    else
    {
        res.setZero();
    }


    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() { param.setZero(); }

  DoubleSphereCamera(const VecN& p) { param = p; }

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // Implementing DS Projection
    Scalar d1 = sqrt((x * x) + (y * y) + (z * z));
    Scalar d2 = sqrt((x * x) + (y * y) + (((xi * d1) + z) * ((xi * d1) + z)));

    Scalar denominator = ((alpha * d2) + ((Scalar(1) - alpha) * ((xi * d1) + z)));

    if (denominator != Scalar(0))
    {
        res[0] = (((fx * x) / denominator) + cx);
        res[1] = (((fy * y) / denominator) + cy);
    }
    else
    {

    }

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    Vec3 res;

    //Implementing DS Unprojection
    Scalar mx = Scalar(0);
    Scalar my = Scalar(0);
    Scalar mz = Scalar(0);

    if(fx != Scalar(0))
    {
        mx = ((p[0] - cx) / fx);
    }
    if(fy != Scalar(0))
    {
        my = ((p[1] - cy) / fy);
    }

    Scalar r_squre = ((mx * mx) + (my * my));

    Scalar denominator1 = ((alpha * sqrt((Scalar(1) - (((Scalar(2) * alpha) - Scalar(1)) * r_squre)))) + (Scalar(1) - alpha));

    if(denominator1 != Scalar(0))
    {
        mz = ((Scalar(1) - (alpha * alpha * r_squre)) / denominator1);
    }

    Scalar denominator2 = (r_squre + (mz * mz));

    if (denominator2 != Scalar(0))
    {
        Scalar numerator = ((mz * xi) + sqrt((mz * mz) + ((Scalar(1) - (xi * xi)) * r_squre)));

        res[0] = ((mx * numerator) / denominator2);
        res[1] = ((my * numerator) / denominator2);
        res[2] = (((mz * numerator) / denominator2) - xi);
    }
    else
    {
        res.setZero();
    }

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() { param.setZero(); }

  KannalaBrandt4Camera(const VecN& p) { param = p; }

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;

    // Implementing KB Projection
    Scalar r = sqrt((x * x) + (y * y));
//    std::cout << "Printing r: " << r << std::endl;
    Scalar theta = atan2(r, z);
    Scalar f_theta =
            (theta +
            (k1 * theta * theta * theta) +
            (k2 * theta * theta * theta * theta * theta) +
            (k3 * theta * theta * theta * theta * theta * theta * theta) +
            (k4 * theta * theta * theta * theta * theta * theta * theta * theta * theta));

    if(r != Scalar(0))
    {
        res[0] = (((fx * f_theta * x) / r) + cx);
        res[1] = (((fy * f_theta * y) / r) + cy);
    }
    else
    {
        res.setZero();
    }

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    Vec3 res;

    //Implementing KB Unprojection

//    std::cout <<"Printing p values, p[0]: "<< p[0] << " p[1]: "<< p[1] << std::endl;
//    std::cout <<"Printing f values, fx: "<< fx << " fy: "<< fy << std::endl;

    Scalar mx = Scalar(0);
    Scalar my = Scalar(0);
    Scalar mz = Scalar(0);

    if(fx != Scalar(0) && p[0] != Scalar(0))
    {
        mx = ((p[0] - cx) / fx);
    }

    if(fy != Scalar(0) && p[1] != Scalar(0))
    {
        my = ((p[1] - cy) / fy);
    }

    Scalar r_squre = ((mx * mx) + (my * my));
    Scalar r = sqrt(r_squre);

    // Calculating theta from this equation --> (d(theta) - r = 0)
    Scalar theta = (M_PI / Scalar(2)); // Initializing with any angle
    Scalar theta_next = Scalar(0);
    Scalar limit = Scalar(1e-14);

    Scalar f_theta = Scalar(0);
    Scalar f_theta_prime = Scalar(0);

    while(true)
    {

        f_theta = (theta +
                   (k1 * theta * theta * theta) +
                   (k2 * theta * theta * theta * theta * theta) +
                   (k3 * theta * theta * theta * theta * theta * theta * theta) +
                   (k4 * theta * theta * theta * theta * theta * theta * theta * theta * theta) - r);

        f_theta_prime = (Scalar(1) +
                   (Scalar(3) * k1 * theta * theta) +
                   (Scalar(5) * k2 * theta * theta * theta * theta) +
                   (Scalar(7) * k3 * theta * theta * theta * theta * theta * theta) +
                   (Scalar(9) * k4 * theta * theta * theta * theta * theta * theta * theta * theta));

        theta_next = (theta - (f_theta / f_theta_prime));
        Scalar diff = Scalar(0);

        if (theta_next > theta)
        {
            diff = theta_next - theta;
        }
        else
        {
            diff = theta - theta_next;
        }

        if(diff < limit)
        {
            break;
        }

        theta = theta_next;
        //std::cout << theta << " " << theta_next << std::endl;
    }


    //std::cout <<"Printing r again: "<< r << std::endl;
    if (r != Scalar(0))
    {
        res[0] = ((mx * sin(theta)) / r);
        res[1] = ((my * sin(theta)) / r);
    }
    else
    {
        res[0] = Scalar(0);
        res[1] = Scalar(0);
    }

    res[2] = cos(theta);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }
};
