#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <vector>

#include "helper.h"
#include "multiStepAPI.h"

namespace faoc {

// Simple logging class (no external dependencies)
class FaocLogger {
 public:
  FaocLogger(const char* file, int line, const char* level) : file_(file), line_(line), level_(level) {}

  ~FaocLogger() { fprintf(stderr, "[%s %s:%d] %s\n", level_, file_, line_, stream_.str().c_str()); }

  template <typename T>
  FaocLogger& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

 private:
  const char* file_;
  int line_;
  const char* level_;
  std::ostringstream stream_;
};

// Logging macro
#define LOG(LEVEL) FaocLogger(__FILE__, __LINE__, #LEVEL)

#ifndef FAOC_MAX_ABSTRACT_DIM_SIZE
#define FAOC_MAX_ABSTRACT_DIM_SIZE 4
#endif

constexpr int kMaxAbstractSetDim = FAOC_MAX_ABSTRACT_DIM_SIZE;
static_assert(kMaxAbstractSetDim > 0, "The maximum abstract action dimension must be positive");

constexpr std::array<double, kMaxAbstractSetDim> CreateArray(double val) {
  std::array<double, kMaxAbstractSetDim> arr{};
  for (int i = 0; i < kMaxAbstractSetDim; ++i) {
    arr[i] = val;
  }
  return arr;  // Returns the array by value
}

constexpr auto kZMin = CreateArray(-1);  // lower bounds on abstract rectangular action set (open limits)
constexpr auto kZMax = CreateArray(1);   // upper bounds on abstract rectangular action set (open limits)

// TODO(Guillem): explore other options
constexpr scale_t kScaleType = NORMALIZE;
constexpr bool kParallel = false;
constexpr int kNThreads = 1;

class ErrorFormatter {
 public:
  ErrorFormatter() = default;
  ~ErrorFormatter() = default;

  template <typename Type>
  ErrorFormatter& operator<<(const Type& value) {
    stream_ << value;
    return *this;
  }

  // We log an error when the formatter is asked to generate a string out of its contents
  std::string ToStr() const {
    LOG(ERROR) << stream_.str();
    return stream_.str();
  }

  // NOLINTNEXTLINE(google-explicit-constructor): clang-tidy wants an `explicit` key here
  operator std::string() const {
    LOG(ERROR) << stream_.str();
    return stream_.str();
  }

  enum ConvertToString { kToStr };
  std::string operator>>(ConvertToString) { return stream_.str(); }

 private:
  std::stringstream stream_;
};

struct MPOnlineSettings {
  double max_opt_time;
  bool hard_online;
  double opt_buffer_time;
  explicit MPOnlineSettings(double max_opt_time = 0, bool hard_online_mode = false, double opt_buffer = 0)
    : max_opt_time(max_opt_time), hard_online(hard_online_mode), opt_buffer_time(opt_buffer) {
    if (max_opt_time < 0) {
      throw std::runtime_error(ErrorFormatter() << "Max optimization time must be >=0 but got " << max_opt_time);
    }
    if (opt_buffer_time < 0) {
      throw std::runtime_error(ErrorFormatter() << "Optimization time buffer must be >=0 but got " << opt_buffer);
    }
    if (hard_online && max_opt_time == 0) {
      throw std::runtime_error(ErrorFormatter() << "Optimization time buffer cannot be zero with hard online mode");
    }
  }
};

struct JointData {
  const std::vector<int> mirroring_logic;
  const std::vector<double> pos_min;
  const std::vector<double> pos_max;
  const std::vector<double> vel_max;
  const std::vector<double> acc_max;
  const std::vector<double> jerk_max;
  explicit JointData(const std::vector<int>& mirroring_logic, const std::vector<double>& pos_min,
                     const std::vector<double>& pos_max, const std::vector<double>& vel_max,
                     const std::vector<double>& acc_max, const std::vector<double>& jerk_max)
    : mirroring_logic(mirroring_logic),
      pos_min(pos_min),
      pos_max(pos_max),
      vel_max(vel_max),
      acc_max(acc_max),
      jerk_max(jerk_max) {
    const size_t n_joints = mirroring_logic.size();
    if (n_joints == 0) {
      throw std::runtime_error("At least one joint must be defined");
    }
    if (pos_min.size() != n_joints || pos_max.size() != n_joints || vel_max.size() != n_joints ||
        acc_max.size() != n_joints || jerk_max.size() != n_joints) {
      throw std::runtime_error("All JointData vectors must have the same length");
    }

    // Check that mirroring logic only contains 1, 0 or -1
    for (const auto& logic : mirroring_logic) {
      if (logic != 1 && logic != 0 && logic != -1) {
        throw std::runtime_error(ErrorFormatter() << "Mirroring logic must be 1, 0, or -1 but got " << logic);
      }
    }
  }
};

inline Eigen::VectorXd make_position_limits_symmetric(mpdata* mpd, int n_joints) {
  Eigen::VectorXd ret(n_joints);
  for (int i = 0; i < n_joints; ++i) {
    ret(i) = (mpd[i].joint_lims.qup + mpd[i].joint_lims.qlow) / 2;
    mpd[i].joint_lims.qup -= ret(i);
    mpd[i].joint_lims.qlow -= ret(i);
  }
  return ret;
}

inline Eigen::VectorXd set_robot_limits(mpdata* mpd, const JointData& joint_data, int n_joints) {
  for (int i = 0; i < n_joints; ++i) {
    mpd[i].joint_lims.qlow = joint_data.pos_min[i];
    mpd[i].joint_lims.qup = joint_data.pos_max[i];
    mpd[i].joint_lims.qdotup = joint_data.vel_max[i];
    mpd[i].joint_lims.qddotup = joint_data.acc_max[i];
    mpd[i].joint_lims.qdddotup = joint_data.jerk_max[i];

    mpd[i].term_state_devs.delta_qlow = 0.0;
    mpd[i].term_state_devs.delta_qup = 0.0;
    mpd[i].term_state_devs.delta_qdotlow = 0.0;
    mpd[i].term_state_devs.delta_qdotup = 0.0;
  }
  return make_position_limits_symmetric(mpd, n_joints);
}

inline bool CheckAllMirrorable(const std::vector<int>& mirroring_logic) {
  for (const auto& logic : mirroring_logic) {
    if (logic != 1 && logic != -1) {
      return false;
    }
  }
  return true;
}

template <uint8_t KJointStateDim>
class FAOC {
  static_assert(KJointStateDim >= 2, "The joint space dimensionality of FAOC must be at least 2");

 public:
  // ZState is used for abstract actions (n_joints x abstract_set_dim)
  using ZState =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Dynamic, kMaxAbstractSetDim>;
  // AbstractRow is used for storing the abstract action of 1 joint (1 x abstract_set_dim)
  using AbstractRow = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, kMaxAbstractSetDim>;
  // XAction is used for storing the abstract action of all joints after the mapping (n_joints x abstract_set_dim)
  using XAction =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Dynamic, kMaxAbstractSetDim>;
  // XState is used for the robot state in the joint space (n_joints x KJointStateDim)
  using XState =
    Eigen::Matrix<double, Eigen::Dynamic, KJointStateDim, Eigen::RowMajor>;  // Robot state in the joint set

  enum class MapToAbstractFailureCodes : int {
    kNotInit = -7,
    kNotInsideControlInvariantSet = -2,
    kActionSetComputationError = -1,
  };

  bool zero_tolerance_setting;

  /// @brief FAOC class constructor for general robots.
  FAOC(std::string faoc_type, const int n_joints, const double tau_c, const int n_l, const uint sampling_freq,
       const int abstract_set_dim, const double max_path_time, JointData joint_data, MPOnlineSettings online_settings)
    : f_s_(sampling_freq),
      tau_s_(1.0 / static_cast<double>(sampling_freq)),
      tau_c_(tau_c),
      n_l_(n_l),
      n_joints_(n_joints),
      abstract_set_dim_(abstract_set_dim),
      max_p_time_(max_path_time),
      hard_online_mode_(online_settings.hard_online),
      max_opt_cycles_(static_cast<uint32_t>(floor(online_settings.max_opt_time * static_cast<double>(f_s_)))),
      max_opt_time_(static_cast<double>(max_opt_cycles_) / static_cast<double>(f_s_)),
      max_buffered_opt_time_(online_settings.max_opt_time - online_settings.opt_buffer_time),
      pos_range_correction_(n_joints),
      time_axis_(ComputeTimeAxis()),
      faoc_type_(std::move(faoc_type)),
      mirroring_available_(CheckAllMirrorable(joint_data.mirroring_logic)),
      mirroring_symmetry_(joint_data.mirroring_logic) {
    if (n_joints_ <= 0) {
      throw std::runtime_error("The number of joints must be at least one");
    }
    if (joint_data.mirroring_logic.size() != static_cast<size_t>(n_joints_)) {
      throw std::runtime_error(ErrorFormatter() << "JointData size (" << joint_data.mirroring_logic.size()
                                                << ") does not match n_joints (" << n_joints_ << ")");
    }
    mpd_.resize(n_joints_);
    pos_range_correction_ = set_robot_limits(mpd_.data(), joint_data, n_joints_);

    FAOCInit();
  }

  void FAOCInit() {
    if (tau_c_ < 0.005) {
      throw std::runtime_error(ErrorFormatter() << "tau_c must be at least 0.005!");
    }
    if (abstract_set_dim_ < 1 || abstract_set_dim_ > kMaxAbstractSetDim) {
      throw std::runtime_error(ErrorFormatter() << "abstract_set_dim must be in [1, " << kMaxAbstractSetDim
                                                << "] but got " << abstract_set_dim_);
    }
    if (n_l_ < abstract_set_dim_) {
      throw std::runtime_error(ErrorFormatter() << "n_l must be greater or equal to the action space dimensionality");
    }
    solver_initialized_ = false;
    reset_planner_initialized_ = false;
    InitializeAbstractStorage();
    CheckIfZeroTolerances();
  }

  /// @brief Sets the gain of the velocity limits (must be larger than 0) used as constraints in the optimization.
  /// @param v_gain A n_joints-length array containing gain coefficients for the joint-wise velocity limits.
  /// @return Success or failure code
  int SetVelocityLimitGain(const Eigen::VectorXd& v_gain) {
    if (solver_initialized_) {
      LOG(ERROR) << "This function is not allowed after the solver has been initialized";
      return EXIT_FAILURE;
    }
    if (v_gain.size() != n_joints_) {
      LOG(ERROR) << "Velocity gain vector must have size " << n_joints_ << " but got " << v_gain.size();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < n_joints_; ++i) {
      if (v_gain[i] <= 0) {
        LOG(ERROR) << "Invalid gain " << v_gain[i] << " for axis " << i + 1 << ". The gain values must be > 0";
        return EXIT_FAILURE;
      }
    }
    for (int i = 0; i < n_joints_; ++i) {
      mpd_[i].joint_lims.qdotup *= v_gain[i];
    }
    return EXIT_SUCCESS;
  }

  /// @brief Sets the gain of the acceleration limits (must be larger than 0) used as constraints in the optimization.
  /// @param a_gain A n_joints-length array containing gain coefficients for the joint-wise acceleration limits.
  /// @return Success or failure code
  int SetAccelerationLimitGain(const Eigen::VectorXd& a_gain) {
    if (solver_initialized_) {
      LOG(ERROR) << "This function is not allowed after the solver has been initialized";
      return EXIT_FAILURE;
    }
    if (a_gain.size() != n_joints_) {
      LOG(ERROR) << "Acceleration gain vector must have size " << n_joints_ << " but got " << a_gain.size();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < n_joints_; ++i) {
      if (a_gain[i] <= 0) {
        LOG(ERROR) << "Invalid gain " << a_gain[i] << " for axis " << i + 1 << ". The gain values must be > 0";
        return EXIT_FAILURE;
      }
    }
    for (int i = 0; i < n_joints_; ++i) {
      mpd_[i].joint_lims.qddotup *= a_gain[i];
    }
    return EXIT_SUCCESS;
  }

  /// @brief Sets the gain of the jerk limits (must be larger than 0) used as constraints in the optimization.
  /// @param a_gain A n_joints-length array containing gain coefficients for the joint-wise jerk limits.
  /// @return Success or failure code
  int SetJerkLimitGain(const Eigen::VectorXd& j_gain) {
    if (solver_initialized_) {
      LOG(ERROR) << "This function is not allowed after the solver has been initialized";
      return EXIT_FAILURE;
    }
    if (j_gain.size() != n_joints_) {
      LOG(ERROR) << "Jerk gain vector must have size " << n_joints_ << " but got " << j_gain.size();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < n_joints_; ++i) {
      if (j_gain[i] <= 0) {
        LOG(ERROR) << "Invalid gain " << j_gain[i] << " for axis " << i + 1 << ". The gain values must be > 0";
        return EXIT_FAILURE;
      }
    }
    for (int i = 0; i < n_joints_; ++i) {
      mpd_[i].joint_lims.qdddotup *= j_gain[i];
    }
    return EXIT_SUCCESS;
  }

  /// @brief Sets the tolerances for position and velocity at the connection points
  /// @param p_tol: A n_joints-length array containing the position tolerances in meters or radians (assumed symmetric)
  /// @param v_tol: A n_joints-length array containing the velocity tolerances in meters or radians per second (assumed
  /// symmetric)
  /// @return Success or failure code
  int SetTolerances(const Eigen::VectorXd& p_tol, const Eigen::VectorXd& v_tol) {
    if (solver_initialized_) {
      LOG(ERROR) << "This function is not allowed after the solver has been initialized";
      return EXIT_FAILURE;
    }
    if (p_tol.size() != n_joints_ || v_tol.size() != n_joints_) {
      LOG(ERROR) << "Tolerance vectors must have size " << n_joints_ << " but got p_tol=" << p_tol.size()
                 << " and v_tol=" << v_tol.size();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < n_joints_; ++i) {
      if (p_tol[i] < 0 || v_tol[i] < 0) {
        LOG(ERROR) << "The tolerances cannot be negative!";
        return EXIT_FAILURE;
      }
    }
    for (int i = 0; i < n_joints_; ++i) {
      mpd_[i].term_state_devs.delta_qup = p_tol[i];
      mpd_[i].term_state_devs.delta_qlow = -p_tol[i];
      mpd_[i].term_state_devs.delta_qdotup = v_tol[i];
      mpd_[i].term_state_devs.delta_qdotlow = -v_tol[i];
    }

    CheckIfZeroTolerances();

    return EXIT_SUCCESS;
  }

  int Initialize(const obj_t obj_func, const int n_threads = 1) {
    if (solver_initialized_) {
      LOG(ERROR) << "Solver is already initialized";
      return EXIT_FAILURE;
    }
    if (n_threads < 1) {
      LOG(ERROR) << "n_threads must be at least 1";
      return EXIT_FAILURE;
    }

    const double ret = CallInitAPI(obj_func, n_threads);

    // Health check and get scaling matrices
    sol_size_ = mpd_[0].N;
    for (int i = 0; i < n_joints_; ++i) {
      if (mpd_[i].N != sol_size_) {
        throw std::runtime_error(ErrorFormatter()
                                 << "Joint " << i + 1 << " has size " << mpd_[i].N << " but expected " << sol_size_);
      }
      scaling_action_[i] = Eigen::Map<AbstractRow>(&mpd_[i].D[0], abstract_set_dim_);
      scaling_x_state_[i] = Eigen::Map<Eigen::Matrix<double, 1, KJointStateDim>>(&mpd_[i].D[0], KJointStateDim);
    }

    if (ret < EXIT_SUCCESS) {
      throw std::runtime_error(ErrorFormatter()
                               << "Initialization failed with return code " << static_cast<int>(ret) << ".");
    }
    solver_initialized_ = true;
    LOG(INFO) << "Initialization of motion planning problem took " << ret * 1e6 << "us.";
    Reset();
    return EXIT_SUCCESS;
  }

  int InitializeResetPlanner(Eigen::VectorXd p_reset_low, Eigen::VectorXd p_reset_up, const int mult = 2,
                             const int add_steps = 3, const int n_threads = 1, const int max_n_l = 200,
                             const bool reset_sync = false) {
    if (p_reset_low.size() != n_joints_ || p_reset_up.size() != n_joints_) {
      LOG(ERROR) << "Reset limits must have size " << n_joints_ << " but got p_reset_low=" << p_reset_low.size()
                 << " and p_reset_up=" << p_reset_up.size();
      return EXIT_FAILURE;
    }

    if (!solver_initialized_) {
      LOG(ERROR) << "Initialize the main FAOC solver first!";
      return EXIT_FAILURE;
    }
    if (n_threads < 1) {
      LOG(ERROR) << "n_threads must be at least 1";
      return EXIT_FAILURE;
    }

    // Normalize reset state ranges and save
    p_reset_low -= pos_range_correction_;
    p_reset_up -= pos_range_correction_;

    // Check lower-higher limit consistency
    if ((p_reset_low.array() > p_reset_up.array()).any()) {
      LOG(ERROR) << "The reset position lower limit should be always <= than the upper limit";
      return EXIT_FAILURE;
    }
    // Check consistency with defined limits
    for (int i = 0; i < n_joints_; ++i) {
      if (p_reset_low(i) < mpd_[i].joint_lims.qlow) {
        LOG(ERROR) << "Lower reset position limit for joint " << i + 1 << " exceeds the minimum position limit";
        return EXIT_FAILURE;
      }
      if (p_reset_up(i) > mpd_[i].joint_lims.qup) {
        LOG(ERROR) << "Upper reset position limit for joint " << i + 1 << " exceeds the maximum position limit";
        return EXIT_FAILURE;
      }
    }

    reset_state_low_ = XState::Zero(n_joints_, KJointStateDim);
    reset_state_low_.col(0) = p_reset_low;
    reset_state_high_ = XState::Zero(n_joints_, KJointStateDim);
    reset_state_high_.col(0) = p_reset_up;

    double init_time;
    LOG(INFO) << "Initializing FAOC reset planner. This may take a few seconds...";
    const int ret_code = CallResetPlanInitAPI(mult, add_steps, n_threads, max_n_l, reset_sync, &init_time);
    reset_planner_initialized_ = ret_code == EXIT_SUCCESS;
    if (reset_planner_initialized_) {
      LOG(INFO) << "Initialization of reset planner problem took " << init_time * 1e6 << "us.";
    } else {
      LOG(ERROR) << "Initialization of reset planner failed with exit code: " << ret_code;
    }
    return ret_code;
  }

  /// @brief MPC class destructor
  ~FAOC() {
    if (solver_initialized_) {
      freeData(mpd_.data(), n_joints_);
    }
  }

  /// @brief Resets the Multistep MPC clearing all local variables.
  void Reset() {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return;
    }

    const std::scoped_lock lock(mutex_);
    for (int i = 0; i < n_joints_; ++i) {
      u_complete_[i].resize(0);
      du_complete_[i].resize(0);
      ddu_complete_[i].resize(0);
    }
    get<0>(reset_plan_).resize(0, n_joints_);
    get<1>(reset_plan_).resize(0, n_joints_);
    get<2>(reset_plan_).resize(0, n_joints_);
    get<3>(reset_plan_).resize(0, n_joints_);

    CleanLastSolution();

    split_idx_ = 0;

    initial_state_defined_ = false;
    solution_available_ = false;
  }

  /// @brief Sets the initial joint state of the robot, which must be in the control invariant set.
  /// @param x_0 The initial joint state. Should be a matrix of size NxX, where N is the number of joints and X is the
  /// state dimension
  /// @return Returns 0 if setting succeeded.
  int SetInitialState(XState x_0) {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return EXIT_FAILURE;
    }
    if (solution_available_) {
      LOG(ERROR) << "Reset the MPC first!";
      return EXIT_FAILURE;
    }

    x_0.col(0) -= pos_range_correction_;

    const std::scoped_lock lock(mutex_);
    if (x_0.rows() != n_joints_) {
      LOG(ERROR) << "Initial state must have " << n_joints_ << " rows but got " << x_0.rows();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < n_joints_; ++i) {
      Eigen::Map<Eigen::RowVectorXd>(&mpd_[i].x_0[0], 1, x_0.cols()) = x_0.row(i);
      for (auto j = 0U; j < KJointStateDim; ++j) {
        mpd_[i].x_0[j] *= scaling_x_state_[i][j];
      }
    }
    UpdateInitialStateCallback();
    for (int i = 0; i < n_joints_; ++i) {
      if (!CallInvariantSetCheckAPI(i)) {
        LOG(ERROR) << "Initial state is not inside the maximum control invariant set (joint " << i << ")!";
        return static_cast<int>(MapToAbstractFailureCodes::kNotInsideControlInvariantSet);
      }
    }

    // Action polytope needs to be recomputed for the new state
    initial_state_defined_ = true;
    std::fill(action_set_computed_.begin(), action_set_computed_.end(), false);
    return EXIT_SUCCESS;
  }

  /// @brief Checks if a state is in the maximum controlled invariant set
  /// @param x_0 The initial joint state. Should be a matrix of size NxX, where N is the number of joints and X is the
  /// state dimension
  /// @return Returns <status_code, vector>. Status is EXIT_SUCCESS on success, otherwise a failure code. The vector
  /// contains one integer per joint: 0 if inside the max controlled invariant set, 1 otherwise.
  std::pair<int, Eigen::VectorXi> CheckStateInInvariantSet(XState x_0) {
    if (x_0.rows() != n_joints_) {
      LOG(ERROR) << "State must have " << n_joints_ << " rows but got " << x_0.rows();
      return {static_cast<int>(MapToAbstractFailureCodes::kActionSetComputationError), Eigen::VectorXi::Zero(0)};
    }
    x_0.col(0) -= pos_range_correction_;
    Eigen::VectorXi result(n_joints_);
    // Check that each joint is in max controlled invariant set
    for (int i = 0; i < n_joints_; ++i) {
      x_0.row(i).array() *= scaling_x_state_[i].array();  // Normalize the state
      result[i] = CallInvariantSetCheckExternalAPI(i, x_0.row(i));
    }
    return {EXIT_SUCCESS, result};
  }

  std::pair<int, Eigen::VectorXd> GetUniqueCentroid(int joint_i) {
    if (!initial_state_defined_) {
      LOG(ERROR) << "Initial state is not defined!";
      return std::pair<int, Eigen::VectorXd>(static_cast<int>(MapToAbstractFailureCodes::kNotInit), Eigen::VectorXd());
    }
    return CallGetUniqueCentroidAPI(joint_i);
  }

  Eigen::MatrixXd MirrorJointState(Eigen::MatrixXd joint_state) {
    if (!mirroring_available_) {
      LOG(ERROR) << "Mirroring not available for the general API of FAOC";
      return Eigen::MatrixXd::Zero(0, 0);
    }
    for (int i = 0; i < n_joints_; ++i) {
      for (int j = 0; j < joint_state.cols(); ++j) {
        joint_state(i, j) *= mirroring_symmetry_[i];
      }
    }
    return joint_state;
  }

  /// @brief Sets an abstract action, maps it to a feasible joint action, and computes the connecting trajectory. Then
  /// automatically sets the next current joint state to be the end of the found trajectory so that this function can
  /// be called recursively without needing to call SetInitialState in between.
  /// @param z_action The abstract action. Should be a matrix of size NxZ, where N is the number of joints and Z is the
  /// abstract action dimension.
  /// @return Returns 0 if succeeded.
  int SetAbstractActionAndSolve(ZState z_action) {
    const int ret = SetAbstractAction(z_action);
    if (ret != EXIT_SUCCESS) {
      return ret;
    }
    return Solve();
  }

  /// @brief Sets an abstract action.
  /// @param z_action The abstract action. Should be a matrix of size NxZ, where N is the number of joints and Z is the
  /// abstract action dimension.
  /// @return Returns 0 if succeeded.
  int SetAbstractAction(ZState z_action) {
    if (!initial_state_defined_) {
      LOG(ERROR) << kSolverCodeMap.at(SolveReturnCodes::kNotInit);
      return static_cast<int>(SolveReturnCodes::kNotInit);
    }
    if (z_action.rows() != n_joints_ || z_action.cols() != abstract_set_dim_) {
      LOG(ERROR) << "Abstract action must be of size " << n_joints_ << "x" << abstract_set_dim_ << " but got "
                 << z_action.rows() << "x" << z_action.cols();
      return static_cast<int>(SolveReturnCodes::kBadAction);
    }

    // Check action is in the admissible range
    for (int i = 0; i < abstract_set_dim_; ++i) {
      if (z_action.col(i).maxCoeff() > kZMax[i] || z_action.col(i).minCoeff() < kZMin[i]) {
        LOG(ERROR) << kSolverCodeMap.at(SolveReturnCodes::kBadAction);
        return static_cast<int>(SolveReturnCodes::kBadAction);
      }
    }
    // Set action
    for (int i = 0; i < n_joints_; ++i) {
      Eigen::Map<Eigen::RowVectorXd>(&mpd_[i].mapd->z_N_l[0], 1, z_action.cols()) = z_action.row(i);
    }
    z_n_l_ph_ = z_action;
    return EXIT_SUCCESS;
  }

  /// @brief Maps a joint action to an abstract action, and then updates the initial state to be the found final action,
  /// so that this function can be called recursively without needing to call SetInitialState in between. This call
  /// does not solve any motion planning problem.
  /// @param x_action The joint action. Should be a matrix of size Nx2, where N is the number of joints. This argument
  /// must have width 2 no matter the order of the planner at the moment.
  /// @return A pair of <success_code, mapped_action>. Success code will be 0 if the mapping succeeded. The
  /// mapped_action will be an abstract action of size NxZ, where N is the number of joints. In case of an unsuccessful
  /// call (success_code != 0), mapped_action will be full of zeroes.
  std::pair<int, ZState> MapToAbstractSet(XAction x_action) {
    if (!initial_state_defined_) {
      LOG(ERROR) << "Initial state is not defined!";
      return std::pair<int, ZState>(static_cast<int>(MapToAbstractFailureCodes::kNotInit), z_zero_state_ph_);
    }
    if (x_action.rows() != n_joints_ || x_action.cols() != abstract_set_dim_) {
      LOG(ERROR) << "Joint action must be of size " << n_joints_ << "x" << abstract_set_dim_ << " but got "
                 << x_action.rows() << "x" << x_action.cols();
      return std::pair<int, ZState>(static_cast<int>(SolveReturnCodes::kBadAction), z_zero_state_ph_);
    }

    x_action.col(0) -= pos_range_correction_;

    for (int i = 0; i < n_joints_; ++i) {
      if (!CallInvariantSetCheckAPI(i)) {
        LOG(ERROR) << "Initial state is not inside the maximum control invariant set (joint " << i << ")!";
        return std::pair<int, ZState>(static_cast<int>(MapToAbstractFailureCodes::kNotInsideControlInvariantSet),
                                      z_zero_state_ph_);
      }
      if (CallComputeActionSetAPI(i) != EXIT_SUCCESS) {
        return std::pair<int, ZState>(static_cast<int>(MapToAbstractFailureCodes::kActionSetComputationError),
                                      z_zero_state_ph_);
      }
    }

    SetJointAction(x_action);

    auto z_res = CallInverseMapAPI();
    if (z_res.first != EXIT_SUCCESS) {
      return z_res;
    }

    // All joints worked
    z_n_l_ph_ = z_res.second;
    return z_res;
  }

  int ComputeResetTrajectory(XState current_state, Eigen::VectorXd reset_pos) {
    if (current_state.rows() != n_joints_ || reset_pos.size() != n_joints_) {
      LOG(ERROR) << "Reset inputs must have " << n_joints_ << " joints";
      return EXIT_FAILURE;
    }

    if (!reset_planner_initialized_) {
      LOG(ERROR) << "Reset planner is not initialized";
      return EXIT_FAILURE;
    }

    current_state.col(0) -= pos_range_correction_;
    reset_pos -= pos_range_correction_;

    if ((reset_pos.array() < reset_state_low_.col(0).array()).any() ||
        (reset_pos.array() > reset_state_high_.col(0).array()).any()) {
      LOG(ERROR) << "Reset state is not within the valid position range for one or more joints\n"
                 << "Max pos: " << reset_state_high_.col(0).transpose() << "\n"
                 << "Min pos: " << reset_state_low_.col(0).transpose() << "\n"
                 << "Query: " << reset_pos;
      return EXIT_FAILURE;
    }

    return CallResetPlanAPI(current_state, reset_pos);
  }

  [[nodiscard]] std::string static GetSolverCodeExplanation(const int code) {
    if (code > 0) {
      return "Found " + std::to_string(code) + " solutions";
    }
    try {
      return kSolverCodeMap.at(static_cast<SolveReturnCodes>(code));
    } catch (const std::out_of_range&) {
      return "Unknown code";
    }
  }

  [[nodiscard]] int GetNJoints() const { return n_joints_; }
  [[nodiscard]] double GetSamplingPeriod() const { return tau_s_; }
  [[nodiscard]] uint32_t GetMaxOptCycles() const { return max_opt_cycles_; }
  [[nodiscard]] uint32_t GetFusionIndex() const { return split_idx_; }
  [[nodiscard]] std::string GetFaocType() const { return faoc_type_; }
  [[nodiscard]] bool IsInit() const { return solution_available_; }
  [[nodiscard]] Eigen::VectorXd GetSolutionU(uint joint_i) const { return u_eigen_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetSolutionDU(uint joint_i) const { return du_eigen_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetSolutionDDU(uint joint_i) const { return ddu_eigen_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetSolutionDDDU(uint joint_i) const { return dddu_eigen_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetCompleteU(uint joint_i) const { return u_complete_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetCompleteDU(uint joint_i) const { return du_complete_[joint_i]; }
  [[nodiscard]] Eigen::VectorXd GetCompleteDDU(uint joint_i) const { return ddu_complete_[joint_i]; }
  [[nodiscard]] virtual Eigen::VectorXd GetTimeAxis(uint joint_i, bool complete) const {
    if ((complete && u_complete_[joint_i].rows() == 0) || (!complete && u_eigen_[joint_i].rows() == 0)) {
      LOG(ERROR) << "[ERROR] Could not calculate the time vector since there is no solution yet available";
      return Eigen::VectorXd::Zero(0);
    }
    if (complete) {
      return time_axis_.head(u_complete_[joint_i].rows()).array();
    }
    return time_axis_.head(u_eigen_[joint_i].rows()).array() + split_idx_ * tau_s_;
  }

  [[nodiscard]] bool IsMirroringAvailable() const { return mirroring_available_; }

  [[nodiscard]] int GetJointSpaceDimension() const { return KJointStateDim; }

  [[nodiscard]] int GetAbstractSpaceDimension() const { return abstract_set_dim_; }

  /// @brief  Returns the step size in number of samples.
  /// @return The step size
  [[nodiscard]] int GetStepSize() const { return sol_size_; }

  // These getters are only for unit-testing purposes
  [[nodiscard]] XState GetCurrentState() {
    for (int i = 0; i < n_joints_; ++i) {
      x_ph_.row(i) = Eigen::Map<Eigen::VectorXd>(&mpd_[i].x_0[0], KJointStateDim);
      x_ph_.row(i).array() /= scaling_x_state_[i].array();
    }
    x_ph_.col(0) += pos_range_correction_;
    return x_ph_;
  }
  [[nodiscard]] XAction GetLastJointAction() const { return y_n_l_ph_; }
  [[nodiscard]] XState GetLastJointState() const { return x_n_l_actual_ph_; }
  [[nodiscard]] ZState GetLastAbstractAction() const { return z_n_l_ph_; }
  [[nodiscard]] std::pair<Eigen::MatrixXd, Eigen::VectorXd> GetMaxControlledInvariantSet(int joint_i) {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0));
    }

    return CallInvariantSetGetAPI(joint_i);
  }

  [[nodiscard]] std::pair<Eigen::MatrixXd, Eigen::VectorXd> GetMaxControlledInvariantSetScaled(int joint_i) {
    if (!solver_initialized_) {
      LOG(ERROR) << "Solver not yet initialized";
      return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0));
    }

    return CallInvariantSetScaledGetAPI(joint_i);
  }

  [[nodiscard]] Eigen::MatrixXd GetPositionLimits() {
    Eigen::MatrixXd result(n_joints_, 2);
    for (int i = 0; i < n_joints_; ++i) {
      result(i, 0) = mpd_[i].joint_lims.qlow;
      result(i, 1) = mpd_[i].joint_lims.qup;
    }
    return result;
  }

  [[nodiscard]] Eigen::VectorXd GetVelocityLimits() {
    Eigen::VectorXd result(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
      result(i, 0) = mpd_[i].joint_lims.qdotup;
    }
    return result;
  }

  [[nodiscard]] Eigen::VectorXd GetAccelerationLimits() {
    Eigen::VectorXd result(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
      result(i, 0) = mpd_[i].joint_lims.qddotup;
    }
    return result;
  }

  [[nodiscard]] virtual Eigen::VectorXd GetReducedPositionLimits() { throw std::runtime_error("Not implemented"); }

  [[nodiscard]] virtual Eigen::VectorXd GetReducedVelocityLimits() { throw std::runtime_error("Not implemented"); }

  [[nodiscard]] std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GetResetPlan() {
    return reset_plan_;
  }

  virtual int ExportResetPlanAsBinary(const std::string&) { throw std::runtime_error("Not implemented"); }

  // These functions are only for unit-testing purposes
  int SetRandomInitialState() {
    if (solution_available_) {
      LOG(ERROR) << "Reset the MPC first!";
      return EXIT_FAILURE;
    }
    const std::scoped_lock lock(mutex_);
    for (int i = 0; i < n_joints_; ++i) {
      do {
        getRandomInitialState(&mpd_[i]);
      } while (!CallInvariantSetCheckAPI(i));
    }

    UpdateInitialStateCallback();
    std::fill(action_set_computed_.begin(), action_set_computed_.end(), false);
    initial_state_defined_ = true;
    return EXIT_SUCCESS;
  }

  void SetRandomAction() {
    // Sample a random action
    for (int i = 0; i < n_joints_; ++i) {
      getRandomAbstractAction(&mpd_[i]);
      z_n_l_ph_.row(i) = Eigen::VectorXd::Map(&mpd_[i].mapd->z_N_l[0], abstract_set_dim_);
    }
  }

  int MapFromAbstractSet() {
    for (int i = 0; i < n_joints_; ++i) {
      if (!CallInvariantSetCheckAPI(i)) {
        LOG(ERROR) << "Initial state is not inside the maximum control invariant set (joint " << i << ")!";
        return static_cast<int>(MapToAbstractFailureCodes::kNotInsideControlInvariantSet);
      }
    }
    return CallMapFromAbstractSetAPI();
  }

  int SetRandomActionAndSolve() {
    SetRandomAction();
    return Solve();
  }

  int Solve() {
    const std::scoped_lock lock(mutex_);
    auto start = std::chrono::high_resolution_clock::now();

    if (!initial_state_defined_) {
      LOG(ERROR) << kSolverCodeMap.at(SolveReturnCodes::kInitialStateNotDefined);
      return static_cast<int>(SolveReturnCodes::kInitialStateNotDefined);
    }

    if (solution_available_) {
      split_idx_ = u_complete_[0].rows();
    }

    auto ret = CallSolveAPI();
    if (ret != SolveReturnCodes::kSuccess) {
      LOG(ERROR) << kSolverCodeMap.at(ret);
      return static_cast<int>(ret);
    }

    if (SolutionUpdate() == EXIT_FAILURE) {
      CleanLastSolution();
      LOG(ERROR) << kSolverCodeMap.at(SolveReturnCodes::kCollision);
      return static_cast<int>(SolveReturnCodes::kCollision);
    }

    auto elapsed = 1e-6 * static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::high_resolution_clock::now() - start)
                                                .count());
    if (hard_online_mode_ && elapsed > max_buffered_opt_time_) {
      CleanLastSolution();
      LOG(ERROR) << kSolverCodeMap.at(SolveReturnCodes::kHardRTFailed) << ". Opt time: " << elapsed << " s";
      return static_cast<int>(SolveReturnCodes::kHardRTFailed);
    }

    solution_available_ = true;

    // Recurse initial state
    // We use x_N_l since the terminal position and speed might have changed due to allowed deviations
    for (int i = 0; i < n_joints_; ++i) {
      for (auto j = 0U; j < KJointStateDim; ++j) {
        mpd_[i].x_0[j] = mpd_[i].x_N_l[j];
      }
      x_n_l_actual_ph_.row(i) = Eigen::Map<Eigen::VectorXd>(&mpd_[i].x_N_l[0], KJointStateDim);
      x_n_l_actual_ph_.row(i).array() /= scaling_x_state_[i].array();
    }
    x_n_l_actual_ph_.col(0) += pos_range_correction_;
    RecurseInitialStateCallback();
    std::fill(action_set_computed_.begin(), action_set_computed_.end(), false);

    return EXIT_SUCCESS;
  }

 protected:
  const uint f_s_;              // Robot sampling frequency
  const double tau_s_;          // Robot sampling time [s]
  const double tau_c_;          // length of single spline interval [s]
  const int n_l_;               // number of intervals within the spline
  const int n_joints_;          // Number of robot joints
  const int abstract_set_dim_;  // Dimensionality of the abstract action set (per each joint)
  const double max_p_time_;     // maximum length of the plan [s]

  // For hard online mode, solutions computed in more than max_opt_time_ are treated as failed solutions
  const bool hard_online_mode_;
  // Upper bound of optimization time cost (in cycles and in seconds respectively)
  const uint32_t max_opt_cycles_;
  const double max_opt_time_;
  // max_opt_time_ minus the allowed time buffer, for saving time for the necessary external post-processing
  const double max_buffered_opt_time_;

  // Position limit correction factor (to make FAOC deal with only symmetric limits)
  Eigen::VectorXd pos_range_correction_;

  // Solution size (in samples, per each joint). Should be constant for all episodes and steps
  int sol_size_;

  std::mutex mutex_;

  // Placeholders
  Eigen::VectorXd u_ph_;

  // Used to return the temporal axis of the solutions
  const Eigen::VectorXd time_axis_;

  // Last problem solutions for U, dU/dt, ddU/dt**2
  std::vector<Eigen::VectorXd> u_eigen_;
  std::vector<Eigen::VectorXd> du_eigen_;
  std::vector<Eigen::VectorXd> ddu_eigen_;
  std::vector<Eigen::VectorXd> dddu_eigen_;

  // Overall solutions for U, dU/dt, ddU/dt**2
  std::vector<Eigen::VectorXd> u_complete_;
  std::vector<Eigen::VectorXd> du_complete_;
  std::vector<Eigen::VectorXd> ddu_complete_;

  // Index where the latest solution starts within the complete solution
  uint32_t split_idx_;

  // Flags
  bool solver_initialized_;
  bool initial_state_defined_;
  bool solution_available_;
  bool reset_planner_initialized_;
  std::vector<bool> action_set_computed_;

  // Protected placeholders
  XAction y_n_l_ph_;
  XState x_n_l_actual_ph_;
  ZState z_zero_state_ph_;

  // Reset planner position limits (in row-major order for easy conversion to the expected reset planner format)
  Eigen::Matrix<double, Eigen::Dynamic, KJointStateDim, Eigen::RowMajor> reset_state_low_;
  Eigen::Matrix<double, Eigen::Dynamic, KJointStateDim, Eigen::RowMajor> reset_state_high_;

  std::vector<mpdata> mpd_;

  // Joint state scaling mat for y_n_l_ph_
  std::vector<AbstractRow> scaling_action_;
  // Joint state scaling mat for x_n_l_actual_ph_
  std::vector<Eigen::Matrix<double, 1, KJointStateDim>> scaling_x_state_;

  // Reset plan
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> reset_plan_;

  enum class SolveReturnCodes : int {
    kNotInit = -7,
    kCollision = -6,
    kInfeasibleAction = -5,
    kInitialStateNotDefined = -4,
    kAbstractToJointMapFailed = -3,
    kBadAction = -2,
    kHardRTFailed = -1,
    kSuccess = 0,
  };

  static inline const std::unordered_map<SolveReturnCodes, std::string> kSolverCodeMap = {
    {SolveReturnCodes::kNotInit, "Initial state is not defined"},
    {SolveReturnCodes::kCollision, "Solution found but unsafe"},
    {SolveReturnCodes::kInfeasibleAction, "Action is infeasible"},
    {SolveReturnCodes::kInitialStateNotDefined, "Initial state is not defined"},
    {SolveReturnCodes::kAbstractToJointMapFailed, "Map from abstract set failed"},
    {SolveReturnCodes::kBadAction, "Input action is not within the valid abstract set limits"},
    {SolveReturnCodes::kHardRTFailed, "Hard real time constraint not satisfied"},
    {SolveReturnCodes::kSuccess, "Successful"}};

  void CleanLastSolution() {
    for (int i = 0; i < n_joints_; ++i) {
      u_eigen_[i] = Eigen::VectorXd::Zero(0);
      du_eigen_[i] = Eigen::VectorXd::Zero(0);
      ddu_eigen_[i] = Eigen::VectorXd::Zero(0);
      dddu_eigen_[i] = Eigen::VectorXd::Zero(0);
    }
  }

  int SolutionUpdate() {
    for (int i = 0; i < n_joints_; ++i) {
      // We don't include the first index of u_zoh since it corresponds to the initial state
      u_eigen_[i] = Eigen::Map<Eigen::VectorXd>(&mpd_[i].u_zoh[0] + 1, sol_size_);
      u_eigen_[i].array() += pos_range_correction_(i);
      du_eigen_[i] = Eigen::Map<Eigen::VectorXd>(&mpd_[i].ud_zoh[0] + 1, sol_size_);
      ddu_eigen_[i] = Eigen::Map<Eigen::VectorXd>(&mpd_[i].udd_zoh[0] + 1, sol_size_);
      dddu_eigen_[i] = Eigen::Map<Eigen::VectorXd>(&mpd_[i].uddd_zoh[0] + 1, sol_size_);

      // Update complete control matrix
      if (split_idx_ == 0) {
        u_complete_[i] = u_eigen_[i];
        du_complete_[i] = du_eigen_[i];
        ddu_complete_[i] = ddu_eigen_[i];
      } else {
        u_ph_.resize(split_idx_ + u_eigen_[i].rows());
        u_ph_ << u_complete_[i], u_eigen_[i];
        u_complete_[i] = u_ph_;

        u_ph_.resize(split_idx_ + du_eigen_[i].rows());
        u_ph_ << du_complete_[i], du_eigen_[i];
        du_complete_[i] = u_ph_;

        u_ph_.resize(split_idx_ + ddu_eigen_[i].rows());
        u_ph_ << ddu_complete_[i], ddu_eigen_[i];
        ddu_complete_[i] = u_ph_;
      }
    }

    return EXIT_SUCCESS;
  }

 private:
  // Either quadratic of cubic
  const std::string faoc_type_;

  // Flag for determining if the current FAOC instance supports mirroring
  const bool mirroring_available_;
  // Mirroring logic:
  // 1 indicates that the joint does not need to be mirrored
  // -1 indicates that the joint needs to be mirrored by flipping its sign
  // 0 indicates that the joint is not symmetric
  const std::vector<int> mirroring_symmetry_;

  // Placeholders
  XState x_ph_;
  ZState z_n_l_ph_;

  virtual double CallInitAPI(const obj_t, const int) { throw std::runtime_error("Not implemented"); }
  virtual void UpdateInitialStateCallback() { throw std::runtime_error("Not implemented"); }
  virtual void RecurseInitialStateCallback() { throw std::runtime_error("Not implemented"); }
  virtual int CallMapFromAbstractSetAPI() { throw std::runtime_error("Not implemented"); }
  virtual bool CallInvariantSetCheckAPI(int) { throw std::runtime_error("Not implemented"); }
  virtual int CallInvariantSetCheckExternalAPI(int, Eigen::Vector<double, KJointStateDim>) {
    throw std::runtime_error("Not implemented");
  }
  virtual std::pair<int, Eigen::VectorXd> CallGetUniqueCentroidAPI(int) { throw std::runtime_error("Not implemented"); }
  virtual int CallComputeActionSetAPI(int) { throw std::runtime_error("Not implemented"); }
  virtual std::pair<Eigen::MatrixXd, Eigen::VectorXd> CallInvariantSetGetAPI(int) {
    throw std::runtime_error("Not implemented");
  }
  virtual std::pair<Eigen::MatrixXd, Eigen::VectorXd> CallInvariantSetScaledGetAPI(int) {
    throw std::runtime_error("Not implemented");
  }
  virtual SolveReturnCodes CallSolveAPI() { throw std::runtime_error("Not implemented"); }
  virtual std::pair<int, ZState> CallInverseMapAPI() { throw std::runtime_error("Not implemented"); }
  virtual void SetJointAction(XAction) { throw std::runtime_error("Not implemented"); }
  virtual int CallResetPlanInitAPI(const int, const int, const int, const int, const bool, double*) {
    throw std::runtime_error("Not implemented");
  }
  virtual int CallResetPlanAPI(XState, Eigen::VectorXd) { throw std::runtime_error("Not implemented"); }

  [[nodiscard]] Eigen::VectorXd ComputeTimeAxis() const {
    // Generate a linspace vector
    const int max_length = ceil(max_p_time_ / tau_s_);
    Eigen::VectorXd t_axis = Eigen::VectorXd::LinSpaced(max_length + 1, 0.0, max_length * tau_s_);
    return t_axis;
  }

  void InitializeAbstractStorage() {
    y_n_l_ph_.resize(n_joints_, abstract_set_dim_);
    y_n_l_ph_.setZero();
    z_zero_state_ph_.resize(n_joints_, abstract_set_dim_);
    z_zero_state_ph_.setZero();
    z_n_l_ph_.resize(n_joints_, abstract_set_dim_);
    z_n_l_ph_.setZero();

    scaling_action_.resize(n_joints_);
    scaling_x_state_.resize(n_joints_);
    action_set_computed_.assign(n_joints_, false);
    u_eigen_.resize(n_joints_);
    du_eigen_.resize(n_joints_);
    ddu_eigen_.resize(n_joints_);
    dddu_eigen_.resize(n_joints_);
    u_complete_.resize(n_joints_);
    du_complete_.resize(n_joints_);
    ddu_complete_.resize(n_joints_);
    reset_state_low_.resize(n_joints_, KJointStateDim);
    reset_state_high_.resize(n_joints_, KJointStateDim);
    x_n_l_actual_ph_.resize(n_joints_, KJointStateDim);
    x_n_l_actual_ph_.setZero();
    x_ph_.resize(n_joints_, KJointStateDim);
    x_ph_.setZero();

    for (auto& scaling_action_i : scaling_action_) {
      scaling_action_i.resize(1, abstract_set_dim_);
      scaling_action_i.setZero();
    }
  }

  void CheckIfZeroTolerances() {
    bool zero_tol = true;

    for (auto& mpd_i : mpd_) {
      if (mpd_i.term_state_devs.delta_qup != 0.0 || mpd_i.term_state_devs.delta_qlow != 0.0 ||
          mpd_i.term_state_devs.delta_qdotup != 0.0 || mpd_i.term_state_devs.delta_qdotlow != 0.0) {
        zero_tol = false;
      }
    }

    zero_tolerance_setting = zero_tol;
  }
};

}  // namespace faoc
