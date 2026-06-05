#include <filesystem>

#include "faoc_cubic.hpp"
#include "gtest/gtest.h"

namespace faoc {
namespace {
JointData GetJointData() {
  JointData robot_data(
    {1, -1, -1, -1, -1, -1, -1, -1}, {-0.925, -0.675, -3.14159, -1.5708, -3.14159, -2.44346095, -2.40855437, -6.28319},
    {0.925, 0.675, 3.14159, 1.5708, 3.14159, 2.44346095, 2.40855437, 6.28319},
    {4., 4., 7.85398, 7.85398, 12.5664, 12.5664, 7.8539816, 31.41593},
    {10.8, 15.7, 39.27, 39.27, 125.66, 125.66, 78.54, 628.32}, {600., 600., 750., 750., 1500., 1500., 1000., 6000.});
  return robot_data;
}
}  // namespace

TEST(PolynomialMPCMusashiFAOC, TestAllXYMusashiFAOCScenarios) {
  return;
  const float tau_c = 0.008;
  const int n_l = 4;
  const auto mpc_ptr =
    std::make_unique<FAOCCubicApprox>(tau_c, n_l, 1000, GetJointData(), MPOnlineSettings(0.001, false), 2);

  std::array<double, 8> zero_tol;
  zero_tol.fill(0);

  mpc_ptr->SetTolerances(zero_tol, zero_tol);
  mpc_ptr->Initialize(obj_t::MAGN);

  const int n_episodes = 1e2;  // number of episodes in test loop
  const int n_steps = 25;      // number of consecutive steps in multi-step MPC (per episode)
  int n_failures = 0;

  // init random number generator
  srand(0);

  double run_time = 0;
  struct timespec start;
  struct timespec end;

  for (int j = 0; j < n_episodes; ++j) {
    mpc_ptr->Reset();
    mpc_ptr->SetRandomInitialState();

    clock_gettime(CLOCK_REALTIME, &start);
    for (int m = 0; m < n_steps; ++m) {
      if (mpc_ptr->SetRandomActionAndSolve() != EXIT_SUCCESS) {
        n_failures++;
        // Set the rest of random actions without solving them to have consistency with the RNG
        for (int i = m + 1; i < n_steps; ++i) {
          mpc_ptr->SetRandomAction();
        }
        break;
      }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    run_time += static_cast<double>(end.tv_sec - start.tv_sec) + static_cast<double>(end.tv_nsec - start.tv_nsec) / 1e9;
  }

  LOG(INFO) << "[INFO] Average runtime per call: " << run_time / (n_episodes * n_steps) * 1e6 << "us";
  LOG(INFO) << "[INFO] Number of successfully completed episodes (of " << n_steps
            << " each): " << n_episodes - n_failures << "/" << n_episodes;

  ASSERT_EQ(n_failures, 0);
}

TEST(PolynomialMPCMusashiFAOC, TestInverseMapXYMusashiFAOC) {
  return;
  const float tau_c = 0.008;
  const int n_l = 4;
  const auto mpc_ptr =
    std::make_unique<FAOCCubicApprox>(tau_c, n_l, 1000, GetJointData(), MPOnlineSettings(0.001, false), 2);

  std::array<double, 8> zero_tol;
  zero_tol.fill(0);

  mpc_ptr->SetTolerances(zero_tol, zero_tol);
  mpc_ptr->Initialize(obj_t::MAGN);

  const int n_episodes = 1e2;       // number of episodes in test loop
  const int n_steps = 25;           // number of consecutive steps in multi-step MPC (per episode)
  const double rel_acc_req = 1e-4;  // required relative accuracy for all abstract points |z_n_l - z_c|_1 > radius

  // init random number generator
  srand(0);

  // main test loop, which also illustrates the usage of the main API function
  int err_action_set_computation = 0;
  int err_map_to_abstract_set = 0;
  int err_not_in_set = 0;
  int rel_acc_not_met = 0;
  const int num_unknown_failures = 0;

  double z_c[2] = {0};  // centroid of rectangle (z_min, z_max)
  z_c[0] = (1.0 / 2) * (kZMin[0] + kZMax[0]);
  z_c[1] = (1.0 / 2) * (kZMin[1] + kZMax[1]);

  struct timespec start;

  for (int j = 0; j < n_episodes; ++j) {
    mpc_ptr->Reset();
    mpc_ptr->SetRandomInitialState();

    for (int m = 0; m < n_steps; ++m) {
      mpc_ptr->SetRandomAction();
      if (mpc_ptr->MapFromAbstractSet()) {
        err_action_set_computation++;
        break;
      }
      clock_gettime(CLOCK_REALTIME, &start);
      const FAOCCubicApprox::ZState z_n_l = mpc_ptr->GetLastAbstractAction();
      const FAOCCubicApprox::XAction x_n_l = mpc_ptr->GetLastJointAction();
      const auto z_map_res = mpc_ptr->MapToAbstractSet(x_n_l);
      if (z_map_res.first ==
          static_cast<int>(FAOCCubicApprox::MapToAbstractFailureCodes::kNotInsideControlInvariantSet)) {
        err_not_in_set++;
        break;
      }
      if (z_map_res.first != EXIT_SUCCESS) {
        err_map_to_abstract_set++;
        continue;
      }

      if ((z_n_l - z_map_res.second).lpNorm<1>() > rel_acc_req) {
        LOG(ERROR) << "Accuracy not met " << (z_n_l - z_map_res.second).lpNorm<1>();
        rel_acc_not_met++;
      }

      mpc_ptr->SetAbstractActionAndSolve(z_n_l);
    }
  }
  ASSERT_EQ(num_unknown_failures, 0);
  ASSERT_EQ(err_not_in_set, 0);
  ASSERT_EQ(err_map_to_abstract_set, 0);
  // One test case is outside of the MCIS by about 2e-9
  ASSERT_EQ(err_action_set_computation, 1);
  ASSERT_EQ(rel_acc_not_met, 0);
}

TEST(PolynomialMPCMusashiFAOC, TestResetPlan) {
  const float tau_c = 0.008;
  const int n_l = 4;
  const auto mpc_ptr =
    std::make_unique<FAOCCubicApprox>(tau_c, n_l, 1000, GetJointData(), MPOnlineSettings(0.001, false), 2);

  std::array<double, 8> lims;
  lims.fill(0.7);
  mpc_ptr->SetVelocityLimitGain(lims);
  mpc_ptr->SetAccelerationLimitGain(lims);

  Eigen::Matrix<double, 8, 1> x_final;
  x_final << 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  mpc_ptr->Initialize(obj_t::MAGN);
  mpc_ptr->InitializeResetPlanner(x_final, x_final);

  FAOCCubicApprox::XState x_current;
  x_current << 0.40867014202263113, 0.0008075326262929252, -0.6568272531326045, -0.2571675539571125,
    -0.5027792092793294, -0.5515846624923402, -0.6483328262996743, 0.9150794268435601, 14.170086436554397,
    -0.04202259404325718, 2.447038799997091, 14.725486458097478, -1.7526636154714559, -1.9758415633195014,
    28.097334708403487, 0.5440839623464054, -3.5728546398240764, -67.57064950003667, -0.07623570205286376,
    -0.18156795421681268, 15.058195107653017, -4.456235453425315, -0.47441312043639816, 125.16904928062105;
  ASSERT_EQ(mpc_ptr->ComputeResetTrajectory(x_current, x_final), EXIT_SUCCESS);
  auto reset_plan = mpc_ptr->GetResetPlan();
  const Eigen::VectorXd last_pos = get<0>(reset_plan).bottomRows(1).transpose();
  const Eigen::VectorXd last_vel = get<1>(reset_plan).bottomRows(1).transpose();
  const Eigen::VectorXd last_acc = get<2>(reset_plan).bottomRows(1).transpose();
  const Eigen::VectorXd last_jerk = get<3>(reset_plan).bottomRows(1).transpose();
  ASSERT_TRUE(last_pos.isApprox(x_final.col(0), 1e-9));
  ASSERT_TRUE(std::abs(last_vel.maxCoeff()) < 1e-7);
  ASSERT_TRUE(std::abs(last_acc.maxCoeff()) < 1e-4);

  ASSERT_TRUE(mpc_ptr->ExportResetPlanAsBinary("/tmp/file.bin") == EXIT_SUCCESS);
  ASSERT_TRUE(mpc_ptr->ExportResetPlanAsBinary("/tmp/file.bin") == EXIT_SUCCESS);
  std::vector<std::array<ResetPlannerExport, 8>> rpd;
  ASSERT_TRUE(mpc_ptr->ImportResetPlanFromBinary("/tmp/file.bin", rpd) == EXIT_SUCCESS);

  // Quick check to verify that the final state is non-zero in position but zero in velocity and acceleration
  ASSERT_EQ(rpd.size(), 2UL);
  ASSERT_NE(rpd[0][0].x_f[0], 0.0);
  ASSERT_EQ(rpd[0][0].x_f[1], 0.0);
  ASSERT_EQ(rpd[0][0].x_f[2], 0.0);
  ASSERT_EQ(rpd[0][0].joint_idx, 0);

  ASSERT_TRUE(std::filesystem::remove("/tmp/file.bin"));
}

}  // namespace faoc
