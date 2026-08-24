using System.Collections.Generic;
using UnityEngine;

namespace TexMotion.Runtime.Motion
{
    public enum SmplxJoint : int
    {
        Pelvis = 0,
        L_Hip = 1,
        R_Hip = 2,
        Spine1 = 3,
        L_Knee = 4,
        R_Knee = 5,
        Spine2 = 6,
        L_Ankle = 7,
        R_Ankle = 8,
        Spine3 = 9,
        L_Foot = 10,
        R_Foot = 11,
        Neck = 12,
        L_Collar = 13,
        R_Collar = 14,
        Head = 15,
        L_Shoulder = 16,
        R_Shoulder = 17,
        L_Elbow = 18,
        R_Elbow = 19,
        L_Wrist = 20,
        R_Wrist = 21,
    }

    public static class SmplxJointDefinitions
    {
        public const int JointCount = 22;

        public static readonly int[] Parents = new int[JointCount]
        {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
        };

        public static readonly string[] JointNames = new string[JointCount]
        {
            "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
            "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
            "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
            "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
        };

        public static readonly Dictionary<SmplxJoint, HumanBodyBones> SmplxToHumanBodyBones = new Dictionary<SmplxJoint, HumanBodyBones>
        {
            { SmplxJoint.Pelvis, HumanBodyBones.Hips },
            { SmplxJoint.L_Hip, HumanBodyBones.LeftUpperLeg },
            { SmplxJoint.R_Hip, HumanBodyBones.RightUpperLeg },
            { SmplxJoint.Spine1, HumanBodyBones.Spine },
            { SmplxJoint.L_Knee, HumanBodyBones.LeftLowerLeg },
            { SmplxJoint.R_Knee, HumanBodyBones.RightLowerLeg },
            { SmplxJoint.Spine2, HumanBodyBones.Chest },
            { SmplxJoint.L_Ankle, HumanBodyBones.LeftFoot },
            { SmplxJoint.R_Ankle, HumanBodyBones.RightFoot },
            { SmplxJoint.Spine3, HumanBodyBones.UpperChest },
            { SmplxJoint.L_Foot, HumanBodyBones.LeftToes },
            { SmplxJoint.R_Foot, HumanBodyBones.RightToes },
            { SmplxJoint.Neck, HumanBodyBones.Neck },
            { SmplxJoint.L_Collar, HumanBodyBones.LeftShoulder },
            { SmplxJoint.R_Collar, HumanBodyBones.RightShoulder },
            { SmplxJoint.Head, HumanBodyBones.Head },
            { SmplxJoint.L_Shoulder, HumanBodyBones.LeftUpperArm },
            { SmplxJoint.R_Shoulder, HumanBodyBones.RightUpperArm },
            { SmplxJoint.L_Elbow, HumanBodyBones.LeftLowerArm },
            { SmplxJoint.R_Elbow, HumanBodyBones.RightLowerArm },
            { SmplxJoint.L_Wrist, HumanBodyBones.LeftHand },
            { SmplxJoint.R_Wrist, HumanBodyBones.RightHand },
        };
    }
}
