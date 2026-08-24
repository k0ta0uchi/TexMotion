using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace TexMotion.Runtime.Native
{
    /// <summary>
    /// Holds raw generated motion output from Kimodo (SMPL-X22).
    /// </summary>
    public class GeneratedMotionData
    {
        public int Frames { get; }
        public int JointCount { get; }
        public Vector3[] RootPositions { get; }
        public Quaternion[,] LocalRotations { get; } // [frame, joint]
        public float FrameRate { get; set; } = 20.0f; // Kimodo standard output frame rate

        public GeneratedMotionData(int frames, int jointCount, Vector3[] rootPositions, Quaternion[,] localRotations, float frameRate = 20.0f)
        {
            Frames = frames;
            JointCount = jointCount;
            RootPositions = rootPositions;
            LocalRotations = localRotations;
            FrameRate = frameRate;
        }
    }

    /// <summary>
    /// Safe, high-level wrapper around the Kimodo native inference engine.
    /// </summary>
    public class KimodoInferenceEngine : IDisposable
    {
        private IntPtr _modelHandle = IntPtr.Zero;
        private readonly object _lock = new object();

        public bool IsModelLoaded => _modelHandle != IntPtr.Zero;

        public bool LoadModel(string motionGgufPath, string textGgufPath, out string errorMessage)
        {
            return LoadModel(motionGgufPath, textGgufPath, null, KimodoDevice.Auto, out errorMessage);
        }

        public bool LoadModel(string motionGgufPath, string textGgufPath, string textAdapterGgufPath, KimodoDevice device, out string errorMessage)
        {
            lock (_lock)
            {
                UnloadModel();

                var options = KimodoRuntimeOptions.CreateDefault(device);
                byte[] errBuffer = new byte[1024];

                _modelHandle = KimodoNativeBinding.kimodo_model_load(
                    motionGgufPath,
                    textGgufPath,
                    textAdapterGgufPath,
                    ref options,
                    errBuffer,
                    errBuffer.Length);

                if (_modelHandle == IntPtr.Zero)
                {
                    errorMessage = Encoding.UTF8.GetString(errBuffer).TrimEnd('\0');
                    if (string.IsNullOrEmpty(errorMessage))
                    {
                        errorMessage = "Unknown error while loading Kimodo model.";
                    }
                    return false;
                }

                errorMessage = null;
                return true;
            }
        }

        public void UnloadModel()
        {
            lock (_lock)
            {
                if (_modelHandle != IntPtr.Zero)
                {
                    KimodoNativeBinding.kimodo_model_free(_modelHandle);
                    _modelHandle = IntPtr.Zero;
                }
            }
        }

        public Task<GeneratedMotionData> GenerateAsync(
            string prompt,
            uint frames = 60,
            uint diffusionSteps = 20,
            ulong seed = 0,
            float textCfg = 2.0f,
            float constraintCfg = 2.0f)
        {
            return Task.Run(() => Generate(prompt, frames, diffusionSteps, seed, textCfg, constraintCfg));
        }

        public GeneratedMotionData Generate(
            string prompt,
            uint frames = 60,
            uint diffusionSteps = 20,
            ulong seed = 0,
            float textCfg = 2.0f,
            float constraintCfg = 2.0f)
        {
            lock (_lock)
            {
                if (!IsModelLoaded)
                {
                    throw new InvalidOperationException("Kimodo model is not loaded. Call LoadModel first.");
                }

                var options = KimodoGenerationOptions.CreateDefault(frames, diffusionSteps, seed, textCfg, constraintCfg);
                byte[] errBuffer = new byte[1024];

                IntPtr motionHandle = KimodoNativeBinding.kimodo_generate(
                    _modelHandle,
                    prompt,
                    ref options,
                    errBuffer,
                    errBuffer.Length);

                if (motionHandle == IntPtr.Zero)
                {
                    string err = Encoding.UTF8.GetString(errBuffer).TrimEnd('\0');
                    if (string.IsNullOrEmpty(err))
                    {
                        IntPtr lastErrPtr = KimodoNativeBinding.kimodo_model_last_error(_modelHandle);
                        if (lastErrPtr != IntPtr.Zero)
                        {
                            err = Marshal.PtrToStringAnsi(lastErrPtr);
                        }
                    }
                    throw new Exception($"Kimodo generation failed: {err}");
                }

                try
                {
                    int totalFrames = KimodoNativeBinding.kimodo_motion_frames(motionHandle);
                    int totalJoints = KimodoNativeBinding.kimodo_motion_joints(motionHandle);

                    IntPtr rotPtr = KimodoNativeBinding.kimodo_motion_local_rotations_xyzw(motionHandle);
                    IntPtr posPtr = KimodoNativeBinding.kimodo_motion_root_positions(motionHandle);

                    // Read float arrays
                    float[] rotFloats = new float[totalFrames * totalJoints * 4];
                    float[] posFloats = new float[totalFrames * 3];

                    Marshal.Copy(rotPtr, rotFloats, 0, rotFloats.Length);
                    Marshal.Copy(posPtr, posFloats, 0, posFloats.Length);

                    Vector3[] rootPositions = new Vector3[totalFrames];
                    Quaternion[,] localRotations = new Quaternion[totalFrames, totalJoints];

                    for (int t = 0; t < totalFrames; t++)
                    {
                        // Root position (Kimodo is right-handed/Z-up or Y-up, map to Unity left-handed)
                        // Kimodo root position: (X, Y, Z)
                        rootPositions[t] = new Vector3(
                            posFloats[t * 3 + 0],
                            posFloats[t * 3 + 1],
                            posFloats[t * 3 + 2]
                        );

                        for (int j = 0; j < totalJoints; j++)
                        {
                            int idx = (t * totalJoints + j) * 4;
                            float x = rotFloats[idx + 0];
                            float y = rotFloats[idx + 1];
                            float z = rotFloats[idx + 2];
                            float w = rotFloats[idx + 3];

                            localRotations[t, j] = new Quaternion(x, y, z, w);
                        }
                    }

                    return new GeneratedMotionData(totalFrames, totalJoints, rootPositions, localRotations);
                }
                finally
                {
                    KimodoNativeBinding.kimodo_motion_free(motionHandle);
                }
            }
        }

        public void Dispose()
        {
            UnloadModel();
            GC.SuppressFinalize(this);
        }

        ~KimodoInferenceEngine()
        {
            UnloadModel();
        }
    }
}
