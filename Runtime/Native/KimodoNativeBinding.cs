using System;
using System.Runtime.InteropServices;

namespace TexMotion.Runtime.Native
{
    public enum KimodoDevice : int
    {
        Auto = 0,
        Cpu = 1,
        Vulkan = 2
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KimodoRuntimeOptions
    {
        public uint size;
        public uint threads;
        public KimodoDevice device;
        [MarshalAs(UnmanagedType.LPStr)]
        public string backend_dir;

        public static KimodoRuntimeOptions CreateDefault(KimodoDevice device = KimodoDevice.Auto, uint threads = 0, string backendDir = null)
        {
            return new KimodoRuntimeOptions
            {
                size = (uint)Marshal.SizeOf(typeof(KimodoRuntimeOptions)),
                threads = threads,
                device = device,
                backend_dir = backendDir
            };
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KimodoGenerationOptions
    {
        public uint size;
        public ulong seed;
        public uint frames;
        public uint diffusion_steps;
        public float text_cfg_weight;
        public float constraint_cfg_weight;

        public static KimodoGenerationOptions CreateDefault(uint frames = 60, uint diffusionSteps = 20, ulong seed = 0, float textCfg = 2.0f, float constraintCfg = 2.0f)
        {
            return new KimodoGenerationOptions
            {
                size = (uint)Marshal.SizeOf(typeof(KimodoGenerationOptions)),
                seed = seed,
                frames = frames,
                diffusion_steps = diffusionSteps,
                text_cfg_weight = textCfg,
                constraint_cfg_weight = constraintCfg
            };
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KimodoEmbedding
    {
        public IntPtr data;
        public uint values;
    }

    /// <summary>
    /// Direct P/Invoke bindings for kimodo.dll C ABI
    /// </summary>
    public static class KimodoNativeBinding
    {
        private const string DllName = "kimodo";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int kimodo_abi_version();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr kimodo_model_load(
            string motion_gguf,
            string text_gguf,
            string text_adapter_gguf,
            ref KimodoRuntimeOptions options,
            [Out] byte[] err,
            int err_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr kimodo_model_load(
            string motion_gguf,
            string text_gguf,
            string text_adapter_gguf,
            IntPtr options,
            [Out] byte[] err,
            int err_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kimodo_model_free(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kimodo_model_last_error(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr kimodo_generate(
            IntPtr model,
            string prompt,
            ref KimodoGenerationOptions options,
            [Out] byte[] err,
            int err_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kimodo_generate_embedding(
            IntPtr model,
            ref KimodoEmbedding embedding,
            ref KimodoGenerationOptions options,
            [Out] byte[] err,
            int err_len);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kimodo_motion_free(IntPtr motion);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int kimodo_motion_frames(IntPtr motion);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int kimodo_motion_joints(IntPtr motion);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kimodo_motion_local_rotations_xyzw(IntPtr motion);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kimodo_motion_root_positions(IntPtr motion);
    }
}
