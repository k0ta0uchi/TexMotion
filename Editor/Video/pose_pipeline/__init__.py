"""
pose_pipeline - TexMotion Modular Pose Extraction Pipeline

Architecture:
- observations: Immutable observation representations and uncertainty metrics.
- backends: Modular detectors (MediaPipe, RTMPose/DWPose ONNX) with preflight checks and fallback.
- kinematics: Anatomical joint limits, bone lengths, forward kinematics, SO(3) converters.
- sequence_fit: Temporal smoothing, multi-hypothesis crossing resolution, contact fitting.
"""

from .observations import (
    ObservationStatus,
    Keypoint2D,
    Keypoint3D,
    UncertaintyInterval,
    FrameObservations
)
from .kinematics import (
    SMPLX_JOINT_COUNT,
    SMPLX_PARENTS,
    SMPLX_JOINT_NAMES,
    SMPLX_DEFAULT_OFFSETS,
    JOINT_LIMITS_DEG,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_slerp,
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_to_quaternion,
    quaternion_to_euler,
    matrix_to_euler,
    forward_kinematics_smplx,
    constrain_knee_flexion,
    swing_twist_decompose,
    apply_anatomical_joint_limits,
    BoneLengthModel
)
from .sequence_fit import SequenceOptimizer
from .vitpose_runner import (
    ViTPoseRunnerError,
    ViTPoseFeatureRunner,
    ViTPoseRunner,
    HMR2FeatureRunner,
    HMR2Runner,
    create_vitpose_runner,
    create_hmr2_runner,
)
from .wham_preprocess import export_vitpose_features_from_video
from .backends import (
    PoseBackend,
    BackendCapabilities,
    MediaPipeBackend,
    RTMPoseBackend,
    PyTorchBackendConfig,
    PyTorchPoseBackend,
    TorchPoseBackend,
    create_pose_backend,
    check_backend_available,
    check_pytorch_backend_available,
    check_mediapipe_available,
    verify_model_manifest
)

__version__ = "2.0.0"

__all__ = [
    "ObservationStatus",
    "Keypoint2D",
    "Keypoint3D",
    "UncertaintyInterval",
    "FrameObservations",
    "SMPLX_JOINT_COUNT",
    "SMPLX_PARENTS",
    "SMPLX_JOINT_NAMES",
    "SMPLX_DEFAULT_OFFSETS",
    "JOINT_LIMITS_DEG",
    "normalize_quaternion",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_slerp",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "matrix_to_euler",
    "forward_kinematics_smplx",
    "constrain_knee_flexion",
    "swing_twist_decompose",
    "apply_anatomical_joint_limits",
    "BoneLengthModel",
    "SequenceOptimizer",
    "ViTPoseRunnerError",
    "ViTPoseFeatureRunner",
    "ViTPoseRunner",
    "HMR2FeatureRunner",
    "HMR2Runner",
    "create_vitpose_runner",
    "create_hmr2_runner",
    "export_vitpose_features_from_video",
    "PoseBackend",
    "BackendCapabilities",
    "MediaPipeBackend",
    "RTMPoseBackend",
    "PyTorchBackendConfig",
    "PyTorchPoseBackend",
    "TorchPoseBackend",
    "create_pose_backend",
    "check_backend_available",
    "check_pytorch_backend_available",
    "check_mediapipe_available",
    "verify_model_manifest",
]
