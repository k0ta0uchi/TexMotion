using System.Collections.Generic;
using UnityEngine;

namespace TexMotion.Runtime.Motion
{
    public enum HandPoseType
    {
        NaturalRelaxed, // 自然に軽く曲げた手 (おすすめ)
        KeepFree,       // 指のアニメーションを含めない (VRChatハンドサインに委ねる)
        Fist,           // 握り拳 (グー)
        OpenPalm,       // しっかり開いた手 (パー)
        Peace,          // ピースサイン
        Point           // 指差し
    }

    public static class HandPosePresets
    {
        public static readonly HumanBodyBones[] LeftFingerBones = new HumanBodyBones[]
        {
            HumanBodyBones.LeftThumbProximal, HumanBodyBones.LeftThumbIntermediate, HumanBodyBones.LeftThumbDistal,
            HumanBodyBones.LeftIndexProximal, HumanBodyBones.LeftIndexIntermediate, HumanBodyBones.LeftIndexDistal,
            HumanBodyBones.LeftMiddleProximal, HumanBodyBones.LeftMiddleIntermediate, HumanBodyBones.LeftMiddleDistal,
            HumanBodyBones.LeftRingProximal, HumanBodyBones.LeftRingIntermediate, HumanBodyBones.LeftRingDistal,
            HumanBodyBones.LeftLittleProximal, HumanBodyBones.LeftLittleIntermediate, HumanBodyBones.LeftLittleDistal
        };

        public static readonly HumanBodyBones[] RightFingerBones = new HumanBodyBones[]
        {
            HumanBodyBones.RightThumbProximal, HumanBodyBones.RightThumbIntermediate, HumanBodyBones.RightThumbDistal,
            HumanBodyBones.RightIndexProximal, HumanBodyBones.RightIndexIntermediate, HumanBodyBones.RightIndexDistal,
            HumanBodyBones.RightMiddleProximal, HumanBodyBones.RightMiddleIntermediate, HumanBodyBones.RightMiddleDistal,
            HumanBodyBones.RightRingProximal, HumanBodyBones.RightRingIntermediate, HumanBodyBones.RightRingDistal,
            HumanBodyBones.RightLittleProximal, HumanBodyBones.RightLittleIntermediate, HumanBodyBones.RightLittleDistal
        };

        /// <summary>
        /// Gets relative local Euler angles (bend rotation) for finger bones based on pose type.
        /// </summary>
        public static Quaternion GetFingerLocalRotation(HandPoseType pose, HumanBodyBones bone, bool isLeft)
        {
            float side = isLeft ? 1f : -1f;

            switch (pose)
            {
                case HandPoseType.NaturalRelaxed:
                    return GetRelaxedRotation(bone, side);

                case HandPoseType.Fist:
                    return GetFistRotation(bone, side);

                case HandPoseType.OpenPalm:
                    return Quaternion.Euler(0, 0, 0);

                case HandPoseType.Peace:
                    return GetPeaceRotation(bone, side);

                case HandPoseType.Point:
                    return GetPointRotation(bone, side);

                default:
                    return Quaternion.identity;
            }
        }

        private static Quaternion GetRelaxedRotation(HumanBodyBones bone, float side)
        {
            string name = bone.ToString();
            if (name.Contains("Thumb"))
            {
                return Quaternion.Euler(10f, side * 15f, side * 12f);
            }
            if (name.Contains("Index"))
            {
                return Quaternion.Euler(0, 0, side * 22f);
            }
            if (name.Contains("Middle"))
            {
                return Quaternion.Euler(0, 0, side * 26f);
            }
            if (name.Contains("Ring"))
            {
                return Quaternion.Euler(0, 0, side * 30f);
            }
            if (name.Contains("Little"))
            {
                return Quaternion.Euler(0, 0, side * 32f);
            }
            return Quaternion.identity;
        }

        private static Quaternion GetFistRotation(HumanBodyBones bone, float side)
        {
            string name = bone.ToString();
            if (name.Contains("Thumb"))
            {
                return Quaternion.Euler(20f, side * 30f, side * 30f);
            }
            // Other fingers curl tightly
            return Quaternion.Euler(0, 0, side * 75f);
        }

        private static Quaternion GetPeaceRotation(HumanBodyBones bone, float side)
        {
            string name = bone.ToString();
            if (name.Contains("Index") || name.Contains("Middle"))
            {
                return Quaternion.Euler(0, side * 5f, 0); // straight
            }
            return GetFistRotation(bone, side); // Thumb, Ring, Little closed
        }

        private static Quaternion GetPointRotation(HumanBodyBones bone, float side)
        {
            string name = bone.ToString();
            if (name.Contains("Index"))
            {
                return Quaternion.identity; // index extended
            }
            return GetFistRotation(bone, side);
        }
    }
}
