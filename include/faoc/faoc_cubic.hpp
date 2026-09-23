#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <span>
#include <vector>

#include "faoc.hpp"
#include "helper.h"
#include "resetAPI.h"

namespace faoc {

// Order of spline (only working with dim = 3, i.e. cubic spline, at the moment)
constexpr size_t kJointSetDim = 3;

// Estimate of the maximum length of the plan in seconds
constexpr double kMaxTimeEst = 5.0;

struct ResetPlannerExport {
  int joint_idx;
  std::array<double, kJointSetDim> x_f;
  std::array<double, kJointSetDim> x_0;
  planner_ret_code planner_ret;
};

// Parameters for maximum control invariant set computation
// TODO(Guillem) expose this struct to API
constexpr maxctrlinvset_params kMCISParams = {.zero_tol = 1e-14,
                                              .primal_tol = 1e-11,
                                              .shift_tol = 1e-8,
                                              .h_rel_tol = 1e-8,
                                              .Hh_abs_tol = 1e-12,
                                              .max_iter = 200,
                                              .n_constr_max = 2000,
                                              .re_method = "convh"};

// Parameters for Motion planner QP
constexpr algo_params kMPAlgParams = {.zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12};

// Parameters for Chebyshev LP (only for POS_VEL)
constexpr algo_params kChebyshevAlgParams = {
  .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12, .eps_prox = 1e1, .eta_prox = 1e-10};

// Parameters for direct Chebyshev LP (only for POS_VEL)
constexpr algo_params kDirectChebyshevAlgParams = {
  .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12, .eps_prox = 1e0, .eta_prox = 1e-10};

// Parameters for LP to compute scale factor beta (only for POS_VEL)
constexpr algo_params kBetaAlgParams = {
  .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12, .eps_prox = 1e1, .eta_prox = 1e-10};

// Parameters for LP to compute action set (only for POS)
constexpr algo_params kPosSetAlgParams = {
  .zero_tol = 1e-12, .primal_tol = 1e-11, .dual_tol = 1e-12, .eps_prox = 2e4, .eta_prox = 1e-10};

class FAOCCubicApprox : public FAOC<kJointSetDim> {
 public:
  FAOCCubicApprox(const double tau_c, const int n_l, const uint sampling_freq, const int n_joints, JointData joint_data,
                  MPOnlineSettings online_settings, const int abstract_set_dim)
    : FAOC<kJointSetDim>("cubic_approx", n_joints, tau_c, n_l, sampling_freq, abstract_set_dim, kMaxTimeEst, joint_data,
                         online_settings) {}

  virtual ~FAOCCubicApprox() {
    if (reset_planner_initialized_) {
      freeResetData(rpd_.data(), n_joints_);
    }
  }

  [[nodiscard]] Eigen::VectorXd GetReducedPositionLimits() {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return Eigen::VectorXd::Zero(n_joints_);
    }
    Eigen::VectorXd result(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
      result(i) = mpd_[i].x_max[0] / mpd_[i].D[0];
    }
    return result;
  }

  [[nodiscard]] Eigen::VectorXd GetReducedVelocityLimits() {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return Eigen::VectorXd::Zero(n_joints_);
    }
    Eigen::VectorXd result(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
      result(i) = mpd_[i].x_max[1] / mpd_[i].D[1];
    }
    return result;
  }

  int ExportResetPlanAsBinary(const std::string& export_fname) {
    std::ofstream out(export_fname, std::ios::binary | std::ios::app);
    if (!out) {
      LOG(ERROR) << "File " << export_fname << " cannot be opened.";
      return EXIT_FAILURE;
    }

    std::vector<ResetPlannerExport> export_d(static_cast<size_t>(n_joints_));
    for (int i = 0; i < n_joints_; ++i) {
      export_d[i].joint_idx = static_cast<int>(i);
      std::ranges::copy(std::span{rpd_[i].x_0, kJointSetDim}, export_d[i].x_0.begin());
      std::ranges::copy(std::span{rpd_[i].x_f, kJointSetDim}, export_d[i].x_f.begin());
      export_d[i].planner_ret = rpd_[i].planner_ret;
    }

    // Write a single scenario plus its joint count.
    int n_scenarios = 1;
    int n_joints = n_joints_;
    out.write(reinterpret_cast<const char*>(&n_scenarios), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_joints), sizeof(int));
    out.write(reinterpret_cast<const char*>(export_d.data()),
              static_cast<std::streamsize>(sizeof(ResetPlannerExport) * export_d.size()));
    out.close();

    return EXIT_SUCCESS;
  }

  /* This function is just for testing purposes */
  static int ImportResetPlanFromBinary(const std::string& import_fname,
                                       std::vector<std::vector<ResetPlannerExport>>& import_d) {
    std::ifstream in(import_fname, std::ios::binary);
    if (!in) {
      LOG(ERROR) << "File " << import_fname << " cannot be opened.";
      return EXIT_FAILURE;
    }

    int n_scenarios;
    int n_joints;

    while (in.peek() != EOF) {
      in.read(reinterpret_cast<char*>(&n_scenarios), sizeof(int));
      if (in.eof()) {
        // in case file ends exactly after last int
        LOG(ERROR) << "Unexpected ending of file!";
        return EXIT_FAILURE;
      }

      if (n_scenarios <= 0) {
        LOG(ERROR) << "Invalid number of scenarios in file: " << n_scenarios;
        return EXIT_FAILURE;
      }

      in.read(reinterpret_cast<char*>(&n_joints), sizeof(int));
      if (in.eof() || n_joints <= 0) {
        LOG(ERROR) << "Invalid number of joints in file: " << n_joints;
        return EXIT_FAILURE;
      }

      // Each scenario is a vector of n_joints reset planner exports.
      for (int i = 0; i < n_scenarios; ++i) {
        std::vector<ResetPlannerExport> one_scenario(static_cast<size_t>(n_joints));
        in.read(reinterpret_cast<char*>(one_scenario.data()),
                static_cast<std::streamsize>(sizeof(ResetPlannerExport) * one_scenario.size()));

        if (in.gcount() != static_cast<std::streamsize>(sizeof(ResetPlannerExport) * one_scenario.size())) {
          LOG(ERROR) << "Incomplete scenario read from file.";
          return EXIT_FAILURE;
        }

        import_d.push_back(one_scenario);
      }
    }

    in.close();
    return EXIT_SUCCESS;
  }

 private:
  struct RuntimeBounds {
    std::vector<c_float> lower;
    std::vector<c_float> upper;
    std::vector<double> e_full;
    map_runtime_bounds bounds{};

    explicit RuntimeBounds(mpdata* mpd)
      : lower(mpd->mapd->N_l + mpd->mapd->dim * (mpd->mapd->N_l - 1) + mpd->n_inf),
        upper(lower.size()),
        e_full(mpd->mapd->dim) {
      mappingUpdateBounds(mpd, &bounds, lower.data(), upper.data(), e_full.data());
    }
  };

  std::vector<rpdata> rpd_;
  bool reset_sync_{};

  int CallResetPlanInitAPI(const int mult, const int add_steps, const int n_threads, const int max_n_l,
                           const bool reset_sync, double* init_time) {
    // Define maximum control horizons valid for all states in respective MCIS
    // Note: If any of the maximum horizon lengths are unknown for the respective setting, then simply
    //       define the corresponding entry as zero. In this case, initResetData() will compute the missing entry.
    std::vector<int> n_l_max(static_cast<size_t>(n_joints_), 0);       // for tau_c discretization
    std::vector<int> n_l_max_mult(static_cast<size_t>(n_joints_), 0);  // for mult*tau_c discretization

    if (rpd_.size() != static_cast<size_t>(n_joints_)) {
      rpd_.resize(static_cast<size_t>(n_joints_));
    }

    // Copy joint limits from FAOC planner
    for (int i = 0; i < n_joints_; i++) {
      rpd_[i].joint_lims = mpd_[i].joint_lims;
    }

    const bool parallel = n_threads > 1;

    const bisect_params k_bs_params_mult = {.dN_max = 2, .add_steps_init = 4, .add_steps = add_steps};

    const double* reset_state_low_ptr = reset_state_low_.data();
    const double* reset_state_high_ptr = reset_state_high_.data();

    reset_sync_ = reset_sync;
    auto reset_mode = reset_sync ? rp_mode::SYNC : rp_mode::DIST;

    const int ret_code =
      initResetData(rpd_.data(), static_cast<int>(kJointSetDim), MAGN, n_joints_, tau_c_, mult, max_n_l, n_l_max.data(),
                    n_l_max_mult.data(), f_s_, kScaleType, parallel, n_threads, reset_state_low_ptr,
                    reset_state_high_ptr, &kMCISParams, &kMPAlgParams, &k_bs_params_mult, reset_mode, init_time);

    // Assume the reset state is always at rest (zero vel and acc)
    for (auto& rpd_i : rpd_) {
      rpd_i.x_f[1] = 0.0;
      rpd_i.x_f[2] = 0.0;
    }

    return ret_code;
  }

  double CallInitAPI(const obj_t obj_func, const int n_threads) {
    double init_time;

    // Load motion planner data. This will generate a n_joints_-length mpd_ array.
    LOG(INFO) << "Initializing FAOCCubicApprox. This may take a few seconds...";
    const bool parallel = n_threads > 1;
    map_t map_type;
    if (abstract_set_dim_ == 1) {
      map_type = map_t::POS;
    } else if (abstract_set_dim_ == 2) {
      map_type = map_t::POS_VEL;
    } else {
      LOG(ERROR) << "FAOCCubicApprox currently supports abstract_set_dim equal to 1 or 2, but got "
                 << abstract_set_dim_;
      return EXIT_FAILURE;
    }

    for (int i = 0; i < n_joints_; ++i) {
      int ret = initMapping(&mpd_[i].mapd, kJointSetDim, n_l_, kZMin.data(), kZMax.data(), map_type, &init_time);
      if (ret < 0) {
        LOG(ERROR) << "Initialization of mapdata struct for joint " << i << " failed with return code " << ret
                   << ". Exiting.";
        return static_cast<double>(ret);
      }
    }

    const int ret = initData(mpd_.data(), obj_func, n_joints_, tau_c_, f_s_, kScaleType, parallel, n_threads,
                             &kMCISParams, &kMPAlgParams, &kChebyshevAlgParams, &kDirectChebyshevAlgParams,
                             &kBetaAlgParams, &kPosSetAlgParams, &init_time);

    if (ret != EXIT_SUCCESS) {
      return static_cast<double>(ret);
    }

    return init_time;
  }

  void UpdateInitialStateCallback() {
    // Set the change of jerk to zero (only relevant for DIFF objective function)
    for (auto& mpd_i : mpd_) {
      mpd_i.udddh_m1 = 0.0;
    }
  }

  void RecurseInitialStateCallback() {
    // We set it even if obj_type != DIFF
    for (auto& mpd_i : mpd_) {
      mpd_i.udddh_m1 = mpd_i.udddh[n_l_ - 1];
    }
  }

  int CallMapFromAbstractSetAPI() {
    int ret;
    for (int i = 0; i < n_joints_; ++i) {
      RuntimeBounds runtime_bounds(&mpd_[i]);
      ret = mapFromAbstractSetWithBounds(mpd_[i].mapd, mpd_[i].n_inf, &runtime_bounds.bounds);
      if (ret < 0) {
        LOG(ERROR) << "Failed in mapping function: " << ret;
        return ret;
      }
      y_n_l_ph_.row(i) = Eigen::VectorXd::Map(&mpd_[i].mapd->y_N_l[0], abstract_set_dim_);
      y_n_l_ph_.row(i).array() /= scaling_action_[i].array();
    }
    y_n_l_ph_.col(0) += pos_range_correction_;
    return EXIT_SUCCESS;
  }

  bool CallInvariantSetCheckAPI(int joint_i) {
    return isInsideMaxCtrlInvSetFast(mpd_[joint_i].H_inf_rm, mpd_[joint_i].h_inf,
                                     const_cast<const double**>(&(mpd_[joint_i].x_0)), 1, 1.0 + 1e-12,
                                     mpd_[joint_i].n_inf, kJointSetDim) == 1;
  }

  int CallInvariantSetCheckExternalAPI(int joint_i, Eigen::Vector<double, kJointSetDim> x_0) {
    const double* ptr = x_0.data();
    const double** double_ptr = &ptr;
    return isInsideMaxCtrlInvSetFast(mpd_[joint_i].H_inf_rm, mpd_[joint_i].h_inf, double_ptr, 1, 1.0 + 1e-12,
                                     mpd_[joint_i].n_inf, kJointSetDim);
  }

  std::pair<int, Eigen::VectorXd> CallGetUniqueCentroidAPI(int joint_i) {
    std::array<double, kMaxAbstractSetDim> y_bar{};
    std::array<double, kMaxAbstractSetDim> e_aux{};
    RuntimeBounds runtime_bounds(&mpd_[joint_i]);
    auto ret = getIntPointActionSetWithBounds(mpd_[joint_i].mapd, mpd_[joint_i].n_inf, &runtime_bounds.bounds,
                                              y_bar.data(), e_aux.data());
    if (ret == OPT_ERR) {
      return std::make_pair(ret, Eigen::VectorXd());
    }
    Eigen::VectorXd y_bar_eigen = Eigen::VectorXd::Map(y_bar.data(), abstract_set_dim_);
    y_bar_eigen.array() /= scaling_action_[joint_i].array();
    return std::make_pair(EXIT_SUCCESS, y_bar_eigen);
  }

  // Approximate FAOC does not use an explicit representation of the action set
  int CallComputeActionSetAPI(int) { return EXIT_SUCCESS; }

  std::pair<Eigen::MatrixXd, Eigen::VectorXd> CallInvariantSetGetAPI(int joint_i) {
    // The position offset breaks the symmetry of the scaled set, so return the one-sided form {x | A x <= b}
    const int n_inf = mpd_[joint_i].n_inf;
    Eigen::MatrixXd h_mat(n_inf, kJointSetDim);
    Eigen::VectorXd h_vec(n_inf);
    for (int j = 0; j < n_inf; ++j) {
      h_mat.row(j) = Eigen::VectorXd::Map(mpd_[joint_i].H_inf[j], kJointSetDim);
      h_mat.row(j).array() *= scaling_x_state_[joint_i].array();
      h_vec(j) = mpd_[joint_i].h_inf[j];
    }
    const Eigen::VectorXd shift = h_mat.col(0) * pos_range_correction_(joint_i);
    Eigen::MatrixXd a_mat(2 * n_inf, kJointSetDim);
    a_mat << h_mat, -h_mat;
    Eigen::VectorXd b_vec(2 * n_inf);
    b_vec << h_vec + shift, h_vec - shift;
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(a_mat, b_vec);
  }

  std::pair<Eigen::MatrixXd, Eigen::VectorXd> CallInvariantSetScaledGetAPI(int joint_i) {
    Eigen::MatrixXd h_a_mat(mpd_[joint_i].n_inf, kJointSetDim);
    Eigen::VectorXd h_b_mat(mpd_[joint_i].n_inf);

    for (int j = 0; j < mpd_[joint_i].n_inf; ++j) {
      h_a_mat.row(j) = Eigen::VectorXd::Map(mpd_[joint_i].H_inf[j], kJointSetDim);
      h_b_mat(j) = mpd_[joint_i].h_inf[j];
    }
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(h_a_mat, h_b_mat);
  }

  SolveReturnCodes CallSolveAPI() {
    // TODO(Guillem): Parallelize this for loop
    for (int i = 0; i < n_joints_; ++i) {
      computeStep(static_cast<void*>(&mpd_[i]));
      y_n_l_ph_.row(i) = Eigen::VectorXd::Map(&mpd_[i].mapd->y_N_l[0], abstract_set_dim_).transpose();
      y_n_l_ph_.row(i).array() /= scaling_action_[i].array();
    }
    y_n_l_ph_.col(0) += pos_range_correction_;

    // Check satisfactory results
    for (int i = 0; i < n_joints_; ++i) {
      if (mpd_[i].planner_ret != TRAJ_OK) {
        LOG(ERROR) << "Motion planner failed for joint " << i << " (error code: " << mpd_[i].planner_ret << ")";
        return SolveReturnCodes::kInfeasibleAction;
      }
    }
    return SolveReturnCodes::kSuccess;
  }

  std::pair<int, ZState> CallInverseMapAPI() {
    // Map to abstract set
    ZState z_res;
    z_res.resize(n_joints_, abstract_set_dim_);
    for (int i = 0; i < n_joints_; ++i) {
      RuntimeBounds runtime_bounds(&mpd_[i]);
      const int res = mapToAbstractSetWithBounds(mpd_[i].mapd, mpd_[i].n_inf, &runtime_bounds.bounds);
      if (res < 0) {
        LOG(ERROR) << "Joint action could not be mapped to abstract action: " << res;
        return std::pair<int, ZState>(res, z_zero_state_ph_);
      }
      z_res.row(i) = Eigen::Map<Eigen::VectorXd>(&mpd_[i].mapd->z_N_l[0], abstract_set_dim_);
    }
    return std::pair<int, ZState>(EXIT_SUCCESS, z_res);
  }

  void SetJointAction(XAction x_action) {
    y_n_l_ph_ = x_action;
    x_action.col(0) -= pos_range_correction_;

    // Set final state in joint space
    for (int i = 0; i < n_joints_; ++i) {
      Eigen::Map<Eigen::RowVectorXd>(&mpd_[i].mapd->y_N_l[0], 1, x_action.cols()) = x_action.row(i);
      for (int j = 0; j < abstract_set_dim_; ++j) {
        mpd_[i].mapd->y_N_l[j] *= scaling_action_[i](0, j);
      }
    }
  }

  int SolutionUpdate() {
    // Collision detection check
    const int result = FAOC<kJointSetDim>::SolutionUpdate();
    for (int i = 0; i < n_joints_; ++i) {
      dddu_eigen_[i] = Eigen::Map<Eigen::VectorXd>(&mpd_[i].uddd_zoh[0] + 1, sol_size_);
    }
    return result;
  }

  int CallResetPlanAPI(XState x_current, Eigen::VectorXd p_reset) {
    const std::scoped_lock lock(mutex_);
    for (int i = 0; i < n_joints_; ++i) {
      rpd_[i].udddh_m1 = 0.0;  // only relevant if obj_type == DIFF and obj_type == MIXED
      rpd_[i].x_f[0] = p_reset[i];
      Eigen::Map<Eigen::RowVectorXd>(&rpd_[i].x_0[0], 1, x_current.cols()) = x_current.row(i);
      for (auto j = 0U; j < kJointSetDim; ++j) {
        rpd_[i].x_0[j] *= scaling_x_state_[i][j];
        rpd_[i].x_f[j] *= scaling_x_state_[i][j];
      }
    }

    // TODO(Guillem): Parallelize reset
    threadpool thpool = nullptr;
    for (auto& rpd_i : rpd_) {
      computeResetTraj(static_cast<void*>(&rpd_i));
    }
    if (reset_sync_) {
      syncTrajs(rpd_.data(), n_joints_, thpool);
    }

    if (const auto result_it = std::ranges::find_if(rpd_, [](auto const& data) { return data.planner_ret != TRAJ_OK; });
        result_it != std::ranges::end(rpd_)) {
      auto const idx = std::ranges::distance(std::ranges::begin(rpd_), result_it);
      LOG(ERROR) << "Reset planner failed for joint " << idx << " (error code: " << result_it->planner_ret << ")";
      return EXIT_FAILURE;
    }

    auto const max_len =
      std::ranges::max_element(rpd_, std::ranges::less{}, [](auto const& data) { return data.N; })->N;

    get<0>(reset_plan_).resize(max_len, n_joints_);
    get<0>(reset_plan_).setZero();
    get<0>(reset_plan_).rowwise() += p_reset.transpose();
    get<1>(reset_plan_).resize(max_len, n_joints_);
    get<1>(reset_plan_).setZero();
    get<2>(reset_plan_).resize(max_len, n_joints_);
    get<2>(reset_plan_).setZero();
    get<3>(reset_plan_).resize(max_len, n_joints_);
    get<3>(reset_plan_).setZero();

    for (int i = 0; i < n_joints_; ++i) {
      // We don't include the first index of u_zoh since it corresponds to the initial state
      get<0>(reset_plan_).col(i).head(rpd_[i].N) = Eigen::Map<Eigen::VectorXd>(rpd_[i].u_zoh + 1, rpd_[i].N);
      get<0>(reset_plan_).col(i).array() += pos_range_correction_[i];
      get<1>(reset_plan_).col(i).head(rpd_[i].N) = Eigen::Map<Eigen::VectorXd>(rpd_[i].du_zoh + 1, rpd_[i].N);
      get<2>(reset_plan_).col(i).head(rpd_[i].N) = Eigen::Map<Eigen::VectorXd>(rpd_[i].ddu_zoh + 1, rpd_[i].N);
      get<3>(reset_plan_).col(i).head(rpd_[i].N) = Eigen::Map<Eigen::VectorXd>(rpd_[i].dddu_zoh + 1, rpd_[i].N);
    }
    return EXIT_SUCCESS;
  }
};

}  // namespace faoc
