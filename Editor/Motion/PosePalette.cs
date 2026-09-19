using System;
using TexMotion.Runtime.Motion;
using UnityEngine;

namespace TexMotion.Editor.Motion
{
    public class PoseSlot
    {
        public string Name = "Empty";
        public bool IsOccupied = false;
        public Vector3 RootPosition = Vector3.zero;
        public Quaternion[] LocalRotations = new Quaternion[SmplxJointDefinitions.JointCount];
    }

    /// <summary>
    /// Manages quick-save slots for poses with blended application and body part masking.
    /// </summary>
    public class PosePalette
    {
        public const int SlotCount = 8;
        private readonly PoseSlot[] _slots = new PoseSlot[SlotCount];

        public PosePalette()
        {
            for (int i = 0; i < SlotCount; i++)
            {
                _slots[i] = new PoseSlot();
            }
        }

        public PoseSlot GetSlot(int index)
        {
            if (index < 0 || index >= SlotCount) return null;
            return _slots[index];
        }

        public bool Capture(int slotIndex, EditableMotionData data, int frame, string customName = null)
        {
            if (data == null || slotIndex < 0 || slotIndex >= SlotCount || frame < 0 || frame >= data.Frames)
                return false;

            var slot = _slots[slotIndex];
            slot.Name = string.IsNullOrEmpty(customName) ? $"Pose {slotIndex + 1} (F{frame})" : customName;
            slot.IsOccupied = true;
            slot.RootPosition = data.GetRootPosition(frame);

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                slot.LocalRotations[j] = data.GetJointRotation(frame, (SmplxJoint)j);
            }

            return true;
        }

        public bool HasPose(int slotIndex)
        {
            if (slotIndex < 0 || slotIndex >= SlotCount) return false;
            return _slots[slotIndex].IsOccupied;
        }

        public bool Clear(int slotIndex)
        {
            return ClearSlot(slotIndex);
        }

        public bool ClearSlot(int slotIndex)
        {
            if (slotIndex < 0 || slotIndex >= SlotCount) return false;
            _slots[slotIndex].IsOccupied = false;
            _slots[slotIndex].Name = "Empty";
            return true;
        }

        public bool Apply(
            int slotIndex,
            EditableMotionData data,
            int frame,
            float blendWeight = 1.0f,
            BodyPartMask mask = BodyPartMask.All,
            bool includeRoot = true)
        {
            if (data == null || slotIndex < 0 || slotIndex >= SlotCount || frame < 0 || frame >= data.Frames)
                return false;

            var slot = _slots[slotIndex];
            if (!slot.IsOccupied) return false;

            blendWeight = Mathf.Clamp01(blendWeight);
            data.RecordUndo($"Apply {slot.Name} ({blendWeight * 100:F0}%) to Frame {frame}");

            if (includeRoot && (mask & BodyPartMask.Pelvis) != 0)
            {
                Vector3 curRoot = data.GetRootPosition(frame);
                data.SetRootPosition(frame, Vector3.Lerp(curRoot, slot.RootPosition, blendWeight));
            }

            int jointCount = SmplxJointDefinitions.JointCount;
            for (int j = 0; j < jointCount; j++)
            {
                SmplxJoint joint = (SmplxJoint)j;
                if (!BodyPartMaskUtility.ContainsJoint(mask, joint)) continue;

                Quaternion curRot = data.GetJointRotation(frame, joint);
                Quaternion targetRot = slot.LocalRotations[j];

                if (Quaternion.Dot(curRot, targetRot) < 0f)
                {
                    targetRot = new Quaternion(-targetRot.x, -targetRot.y, -targetRot.z, -targetRot.w);
                }

                Quaternion blended = Quaternion.Slerp(curRot, targetRot, blendWeight);
                data.SetJointRotation(frame, joint, blended);
            }

            return true;
        }
    }
}
