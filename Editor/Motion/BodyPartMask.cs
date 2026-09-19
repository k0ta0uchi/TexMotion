using System;
using TexMotion.Runtime.Motion;

namespace TexMotion.Editor.Motion
{
    /// <summary>
    /// Bitmask defining anatomical body part groupings for selective pose operations.
    /// </summary>
    [Flags]
    public enum BodyPartMask
    {
        None     = 0,
        Pelvis   = 1 << 0,
        Spine    = 1 << 1, // Spine1, Spine2, Spine3
        Head     = 1 << 2, // Neck, Head
        LeftArm  = 1 << 3, // L_Collar, L_Shoulder, L_Elbow, L_Wrist
        RightArm = 1 << 4, // R_Collar, R_Shoulder, R_Elbow, R_Wrist
        LeftLeg  = 1 << 5, // L_Hip, L_Knee, L_Ankle, L_Foot
        RightLeg = 1 << 6, // R_Hip, R_Knee, R_Ankle, R_Foot

        // Composite presets
        Arms      = LeftArm | RightArm,
        Legs      = LeftLeg | RightLeg,
        UpperBody = Spine | Head | LeftArm | RightArm,
        LowerBody = Pelvis | LeftLeg | RightLeg,
        All       = Pelvis | Spine | Head | LeftArm | RightArm | LeftLeg | RightLeg
    }

    /// <summary>
    /// Easing curve algorithms for range interpolation / tweening.
    /// </summary>
    public enum EasingType
    {
        Linear,
        EaseIn,
        EaseOut,
        EaseInOut,
        SmoothStep
    }

    public static class BodyPartMaskUtility
    {
        public static bool ContainsJoint(BodyPartMask mask, SmplxJoint joint)
        {
            if (mask == BodyPartMask.All) return true;
            if (mask == BodyPartMask.None) return false;

            BodyPartMask jointCategory = GetJointCategory(joint);
            return (mask & jointCategory) != 0;
        }

        public static BodyPartMask GetJointCategory(SmplxJoint joint)
        {
            switch (joint)
            {
                case SmplxJoint.Pelvis:
                    return BodyPartMask.Pelvis;

                case SmplxJoint.Spine1:
                case SmplxJoint.Spine2:
                case SmplxJoint.Spine3:
                    return BodyPartMask.Spine;

                case SmplxJoint.Neck:
                case SmplxJoint.Head:
                    return BodyPartMask.Head;

                case SmplxJoint.L_Collar:
                case SmplxJoint.L_Shoulder:
                case SmplxJoint.L_Elbow:
                case SmplxJoint.L_Wrist:
                    return BodyPartMask.LeftArm;

                case SmplxJoint.R_Collar:
                case SmplxJoint.R_Shoulder:
                case SmplxJoint.R_Elbow:
                case SmplxJoint.R_Wrist:
                    return BodyPartMask.RightArm;

                case SmplxJoint.L_Hip:
                case SmplxJoint.L_Knee:
                case SmplxJoint.L_Ankle:
                case SmplxJoint.L_Foot:
                    return BodyPartMask.LeftLeg;

                case SmplxJoint.R_Hip:
                case SmplxJoint.R_Knee:
                case SmplxJoint.R_Ankle:
                case SmplxJoint.R_Foot:
                    return BodyPartMask.RightLeg;

                default:
                    return BodyPartMask.None;
            }
        }

        public static float EvaluateEasing(EasingType easing, float t)
        {
            t = UnityEngine.Mathf.Clamp01(t);
            switch (easing)
            {
                case EasingType.Linear:
                    return t;
                case EasingType.EaseIn:
                    return t * t;
                case EasingType.EaseOut:
                    return 1f - (1f - t) * (1f - t);
                case EasingType.EaseInOut:
                    return t < 0.5f ? 2f * t * t : 1f - UnityEngine.Mathf.Pow(-2f * t + 2f, 2f) / 2f;
                case EasingType.SmoothStep:
                    return t * t * (3f - 2f * t);
                default:
                    return t;
            }
        }
    }
}
