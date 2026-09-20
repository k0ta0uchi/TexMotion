import sys
sys.stdout.reconfigure(encoding="utf-8")
import subprocess
import os
import glob

csc = r"C:\Program Files\dotnet\sdk\9.0.318\Roslyn\bincore\csc.dll"
mono_api = r"C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Data\MonoBleedingEdge\lib\mono\4.7.1-api"
managed = r"C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Data\Managed"
unity_engine = r"C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Data\Managed\UnityEngine"

rsp_lines = [
    "-target:exe",
    "-out:stylized_test_runner.exe",
    "-noconfig",
    "-nostdlib",
    "-d:UNITY_EDITOR",
    "-nowarn:0169,0414,0649"
]

for d in glob.glob(os.path.join(mono_api, "*.dll")):
    rsp_lines.append(f'-r:"{d}"')
for d in glob.glob(os.path.join(mono_api, "Facades", "*.dll")):
    rsp_lines.append(f'-r:"{d}"')
for d in glob.glob(os.path.join(managed, "*.dll")):
    rsp_lines.append(f'-r:"{d}"')
for d in glob.glob(os.path.join(unity_engine, "*.dll")):
    rsp_lines.append(f'-r:"{d}"')

# Runtime files
for f in glob.glob(r"C:\Workspace\TexMotion\Runtime\**\*.cs", recursive=True):
    rsp_lines.append(f'"{f}"')

# Required Editor files for Stylized & Physics Tests
editor_files = [
    r"C:\Workspace\TexMotion\Editor\Motion\EditableMotionData.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\MotionIkUtility.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\ContactConstraintSolver.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\AnimationClipBuilder.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\BodyPartMask.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\GlitchDetector.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\StylizedMotionPolisher.cs",
    r"C:\Workspace\TexMotion\Editor\Motion\PenetrationConstraintSolver.cs",
    r"C:\Workspace\TexMotion\Editor\VRChat\PhysBoneConflictDetector.cs",
    r"C:\Workspace\TexMotion\tests\StylizedMotionPolisherTests.cs"
]
for f in editor_files:
    rsp_lines.append(f'"{f}"')

rsp_path = r"C:\Workspace\TexMotion\stylized_test.rsp"
with open(rsp_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rsp_lines))

print("Compiling stylized_test_runner.exe...")
res = subprocess.run(["dotnet", "exec", csc, f"@{rsp_path}"], capture_output=True)
stdout = res.stdout.decode("utf-8", errors="replace")
if res.returncode != 0:
    print("Compile failed:\n", stdout[:2000])
    if os.path.exists(rsp_path):
        os.remove(rsp_path)
    sys.exit(1)
else:
    print("Compilation succeeded! Running tests via Mono...")
    mono_exe = r"C:\Program Files\Unity\Hub\Editor\2022.3.22f1\Editor\Data\MonoBleedingEdge\bin\mono.exe"
    
    env = os.environ.copy()
    mono_paths = [
        managed,
        unity_engine,
        mono_api,
        os.path.join(mono_api, "Facades")
    ]
    env["MONO_PATH"] = ";".join(mono_paths)

    run_res = subprocess.run([mono_exe, "stylized_test_runner.exe"], capture_output=True, env=env)
    out = run_res.stdout.decode("utf-8", errors="replace")
    err = run_res.stderr.decode("utf-8", errors="replace")
    print(out)
    if err:
        print("STDERR:\n", err)
    
    if os.path.exists(rsp_path):
        os.remove(rsp_path)
    if os.path.exists("stylized_test_runner.exe"):
        os.remove("stylized_test_runner.exe")

    sys.exit(run_res.returncode)
