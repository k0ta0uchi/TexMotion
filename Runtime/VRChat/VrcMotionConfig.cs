using System;
using UnityEngine;

namespace TexMotion.Runtime.VRChat
{
    public enum VrcSetupMode
    {
        ModularAvatar,   // Modular Avatar (非破壊・推奨)
        DirectVRCSDK     // VRCExpressionsMenu & VRCAvatarDescriptor (直接追加)
    }

    public enum VrcMotionType
    {
        OneShotEmote, // 1回再生して元のポーズに戻る (Trigger / Button)
        ToggleLoopPose // トグルでON/OFFを切り替えてループ再生 (Toggle)
    }

    public enum VrcTargetLayer
    {
        ActionLayer, // Action Layer (全身の動きを上書き)
        FXLayer      // FX Layer (簡易モーションやアクセサリ同期)
    }

    [System.Serializable]
    public class VrcMotionConfig
    {
        public string MotionName = "CustomEmote";
        public VrcSetupMode SetupMode = VrcSetupMode.ModularAvatar;
        public VrcMotionType MotionType = VrcMotionType.OneShotEmote;
        public VrcTargetLayer TargetLayer = VrcTargetLayer.ActionLayer;
        public string MenuCategory = "TexMotion";
        public bool InPlace = true;
    }
}
