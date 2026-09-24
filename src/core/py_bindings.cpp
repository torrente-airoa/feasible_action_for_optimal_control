#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "faoc_cubic.hpp"

namespace faoc {
namespace py = pybind11;

PYBIND11_MODULE(faoc, m) {
  py::class_<MPOnlineSettings>(m, "MPOnlineSettings")
    .def(py::init<double, bool, double>(), py::arg("max_opt_time") = 0.0, py::arg("hard_online_mode") = false,
         py::arg("opt_buffer") = 0.0)
    .def_readwrite("max_opt_time", &MPOnlineSettings::max_opt_time)
    .def_readwrite("hard_online_mode", &MPOnlineSettings::hard_online)
    .def_readwrite("opt_buffer", &MPOnlineSettings::opt_buffer_time);
  py::class_<JointData>(m, "JointData")
    .def(py::init<const std::vector<int> &, const std::vector<double> &, const std::vector<double> &,
                  const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>(),
         py::arg("joint_mirroring"), py::arg("joint_pos_min"), py::arg("joint_pos_max"), py::arg("joint_vel_max"),
         py::arg("joint_acc_max"), py::arg("joint_jerk_max"))
    .def_readonly("mirroring_logic", &JointData::mirroring_logic)
    .def_readonly("pos_min", &JointData::pos_min)
    .def_readonly("pos_max", &JointData::pos_max)
    .def_readonly("vel_max", &JointData::vel_max)
    .def_readonly("acc_max", &JointData::acc_max)
    .def_readonly("jerk_max", &JointData::jerk_max);
  py::enum_<obj_t>(m, "ObjectiveFunction")
    .value("magnitude", obj_t::MAGN)
    .value("difference", obj_t::DIFF)
    .value("mixed", obj_t::MIXED);
  py::class_<FAOCCubicApprox, std::unique_ptr<FAOCCubicApprox>>(m, "CubicSpline")
    .def(py::init<double, int, uint, int, JointData, MPOnlineSettings, int>(), py::arg("tau_c"), py::arg("n_l"),
         py::arg("sampling_freq"), py::arg("n_joints"), py::arg("joint_data"), py::arg("online_settings"),
         py::arg("abstract_set_dim"))
    .def("initialize", &FAOCCubicApprox::Initialize, py::arg("obj_func"), py::arg("n_threads") = 1)
    .def("initialize_reset_planner", &FAOCCubicApprox::InitializeResetPlanner, py::arg("p_reset_low"),
         py::arg("p_reset_up"), py::arg("mult") = 2, py::arg("add_steps") = 3, py::arg("n_threads") = 1,
         py::arg("max_n_l") = 200, py::arg("reset_sync") = false)
    .def("set_tolerances", &FAOCCubicApprox::SetTolerances, py::arg("p_tol"), py::arg("v_tol"))
    .def("set_velocity_limit_gain", &FAOCCubicApprox::SetVelocityLimitGain, py::arg("v_gain"))
    .def("set_acceleration_limit_gain", &FAOCCubicApprox::SetAccelerationLimitGain, py::arg("a_gain"))
    .def("set_jerk_limit_gain", &FAOCCubicApprox::SetJerkLimitGain, py::arg("j_gain"))
    .def("set_initial_state", &FAOCCubicApprox::SetInitialState, py::arg("x_0"))
    .def("check_state_in_invariant_set", &FAOCCubicApprox::CheckStateInInvariantSet, py::arg("x_0"))
    .def("set_abstract_action_and_solve", &FAOCCubicApprox::SetAbstractActionAndSolve, py::arg("z_action"))
    .def("set_abstract_action", &FAOCCubicApprox::SetAbstractAction, py::arg("z_action"))
    .def("solve", &FAOCCubicApprox::Solve)
    .def("map_to_abstract_set", &FAOCCubicApprox::MapToAbstractSet, py::arg("x_action"))
    .def("map_from_abstract_set", &FAOCCubicApprox::MapFromAbstractSet)
    .def("mirror_joint_state", &FAOCCubicApprox::MirrorJointState, py::arg("x_state_or_action"))
    .def("compute_reset_trajectory", &FAOCCubicApprox::ComputeResetTrajectory, py::arg("current_state"),
         py::arg("reset_pos"))
    .def("reset", &FAOCCubicApprox::Reset)
    .def("export_reset_plan_as_binary", &FAOCCubicApprox::ExportResetPlanAsBinary, py::arg("export_fname"))
    .def("get_u_sol", &FAOCCubicApprox::GetSolutionU, py::arg("joint_i"))
    .def("get_du_sol", &FAOCCubicApprox::GetSolutionDU, py::arg("joint_i"))
    .def("get_ddu_sol", &FAOCCubicApprox::GetSolutionDDU, py::arg("joint_i"))
    .def("get_dddu_sol", &FAOCCubicApprox::GetSolutionDDDU, py::arg("joint_i"))
    .def("get_complete_u", &FAOCCubicApprox::GetCompleteU, py::arg("joint_i"))
    .def("get_complete_du", &FAOCCubicApprox::GetCompleteDU, py::arg("joint_i"))
    .def("get_complete_ddu", &FAOCCubicApprox::GetCompleteDDU, py::arg("joint_i"))
    .def("get_reset_plan", &FAOCCubicApprox::GetResetPlan)
    .def("get_time_axis", &FAOCCubicApprox::GetTimeAxis, py::arg("joint_i"), py::arg("complete"))
    .def("get_last_joint_action", &FAOCCubicApprox::GetLastJointAction)
    .def("get_last_joint_state", &FAOCCubicApprox::GetLastJointState)
    .def("get_current_state", &FAOCCubicApprox::GetCurrentState)
    .def("get_fusion_index", &FAOCCubicApprox::GetFusionIndex)
    .def("get_max_controlled_invariant_set", &FAOCCubicApprox::GetMaxControlledInvariantSet, py::arg("joint_i"))
    .def("get_max_controlled_invariant_set_scaled", &FAOCCubicApprox::GetMaxControlledInvariantSetScaled,
         py::arg("joint_i"))
    .def("get_unique_centroid", &FAOCCubicApprox::GetUniqueCentroid, py::arg("joint_i"))
    .def("get_faoc_type", &FAOCCubicApprox::GetFaocType)
    .def("is_mirroring_available", &FAOCCubicApprox::IsMirroringAvailable)
    .def("get_step_size", &FAOCCubicApprox::GetStepSize)
    .def("get_n_joints", &FAOCCubicApprox::GetNJoints)
    .def("get_abstract_space_dimension", &FAOCCubicApprox::GetAbstractSpaceDimension)
    .def("get_joint_space_dimension", &FAOCCubicApprox::GetJointSpaceDimension)
    .def("get_sampling_period", &FAOCCubicApprox::GetSamplingPeriod)
    .def("get_max_opt_cycles", &FAOCCubicApprox::GetMaxOptCycles)
    .def("get_reduced_position_limits", &FAOCCubicApprox::GetReducedPositionLimits)
    .def("get_reduced_velocity_limits", &FAOCCubicApprox::GetReducedVelocityLimits)
    .def("get_position_limits", &FAOCCubicApprox::GetPositionLimits)
    .def("get_velocity_limits", &FAOCCubicApprox::GetVelocityLimits)
    .def("get_acceleration_limits", &FAOCCubicApprox::GetAccelerationLimits)
    .def_static("get_solver_code_explanation", &FAOCCubicApprox::GetSolverCodeExplanation, py::arg("code"))
    .def("is_init", &FAOCCubicApprox::IsInit)
    .def_readonly("is_zero_tolerance", &FAOCCubicApprox::zero_tolerance_setting);
}

}  // namespace faoc
