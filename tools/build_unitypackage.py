import os
import re
import io
import time
import tarfile

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
avatar_assets_dir = r"C:\Users\k0ta0\AppData\Local\VRChatCreatorCompanion\VRChatProjects\Avatar\Assets"
texmotion_assets_dir = os.path.join(avatar_assets_dir, "TexMotion")
output_package = os.path.join(workspace_root, "TexMotion.unitypackage")

EXCLUDE_DIRS = {
    ".git",
    ".gemini_workflow",
    "scratch",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".uv",
    ".texmotion-venv",
    ".texmotion-uv-cache",
    "Library",
    "Temp",
    "Obj",
    "Build",
    "Builds",
    "Logs",
}

EXCLUDE_FILES = {
    ".gitignore",
    "TexMotion.unitypackage",
    "unity_export.log",
}

def get_guid(meta_path):
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    m = re.search(r"guid:\s*([0-9a-fA-F]{32})", content)
    return m.group(1) if m else None

entries = []

# 1. Root package entry
entries.append({
    "guid": "8a31a1c18a177b6fe15c37b10c622f78",
    "pathname": "Packages/com.k0ta0uchi.texmotion",
    "meta_path": None,
    "asset_path": None,
    "is_dir": True,
})

# 2. Workspace root -> Packages/com.k0ta0uchi.texmotion
for root, dirs, files in os.walk(workspace_root):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    rel_root = os.path.relpath(root, workspace_root)
    
    if rel_root != ".":
        norm_rel = rel_root.replace("\\", "/")
        pkg_path = f"Packages/com.k0ta0uchi.texmotion/{norm_rel}"
        meta_path = root + ".meta"
        guid = get_guid(meta_path)
        if guid:
            entries.append({
                "guid": guid,
                "pathname": pkg_path,
                "meta_path": meta_path,
                "asset_path": None,
                "is_dir": True,
            })
        else:
            print(f"Warning: Directory missing meta: {pkg_path} ({meta_path})")

    for f in files:
        if f.endswith(".meta") or f.endswith(".pyc") or f.endswith(".log") or f.endswith(".tmp"):
            continue
        if f in EXCLUDE_FILES or f.startswith("."):
            continue
        
        file_path = os.path.join(root, f)
        if rel_root == ".":
            norm_rel = f
        else:
            norm_rel = (rel_root + "/" + f).replace("\\", "/")
        
        pkg_path = f"Packages/com.k0ta0uchi.texmotion/{norm_rel}"
        meta_path = file_path + ".meta"
        guid = get_guid(meta_path)
        if guid:
            entries.append({
                "guid": guid,
                "pathname": pkg_path,
                "meta_path": meta_path,
                "asset_path": file_path,
                "is_dir": False,
            })
        else:
            print(f"Warning: File missing meta: {pkg_path} ({meta_path})")

# 3. Assets/TexMotion
if os.path.isdir(texmotion_assets_dir):
    meta_path = texmotion_assets_dir + ".meta"
    guid = get_guid(meta_path)
    if guid:
        entries.append({
            "guid": guid,
            "pathname": "Assets/TexMotion",
            "meta_path": meta_path,
            "asset_path": None,
            "is_dir": True,
        })
    
    for root, dirs, files in os.walk(texmotion_assets_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        rel_root = os.path.relpath(root, texmotion_assets_dir)
        
        if rel_root != ".":
            norm_rel = rel_root.replace("\\", "/")
            pkg_path = f"Assets/TexMotion/{norm_rel}"
            meta_path = root + ".meta"
            guid = get_guid(meta_path)
            if guid:
                entries.append({
                    "guid": guid,
                    "pathname": pkg_path,
                    "meta_path": meta_path,
                    "asset_path": None,
                    "is_dir": True,
                })
        
        for f in files:
            if f.endswith(".meta") or f.endswith(".pyc") or f.endswith(".log") or f.endswith(".tmp"):
                continue
            if f in EXCLUDE_FILES or f.startswith("."):
                continue
            
            file_path = os.path.join(root, f)
            if rel_root == ".":
                norm_rel = f
            else:
                norm_rel = (rel_root + "/" + f).replace("\\", "/")
            
            pkg_path = f"Assets/TexMotion/{norm_rel}"
            meta_path = file_path + ".meta"
            guid = get_guid(meta_path)
            if guid:
                entries.append({
                    "guid": guid,
                    "pathname": pkg_path,
                    "meta_path": meta_path,
                    "asset_path": file_path,
                    "is_dir": False,
                })

print(f"Total entries to package: {len(entries)}")

now = int(time.time())

with tarfile.open(output_package, "w:gz") as tf:
    for e in entries:
        guid = e["guid"]
        pathname_bytes = e["pathname"].encode("utf-8")
        
        dir_info = tarfile.TarInfo(name=guid)
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o777
        dir_info.mtime = now
        tf.addfile(dir_info)
        
        pn_info = tarfile.TarInfo(name=f"{guid}/pathname")
        pn_info.type = tarfile.REGTYPE
        pn_info.mode = 0o777
        pn_info.mtime = now
        pn_info.size = len(pathname_bytes)
        tf.addfile(pn_info, io.BytesIO(pathname_bytes))
        
        if e["meta_path"] and os.path.isfile(e["meta_path"]):
            with open(e["meta_path"], "rb") as mf:
                meta_bytes = mf.read()
            meta_info = tarfile.TarInfo(name=f"{guid}/asset.meta")
            meta_info.type = tarfile.REGTYPE
            meta_info.mode = 0o777
            meta_info.mtime = now
            meta_info.size = len(meta_bytes)
            tf.addfile(meta_info, io.BytesIO(meta_bytes))
        
        if e["asset_path"] and os.path.isfile(e["asset_path"]):
            with open(e["asset_path"], "rb") as af:
                asset_bytes = af.read()
            asset_info = tarfile.TarInfo(name=f"{guid}/asset")
            asset_info.type = tarfile.REGTYPE
            asset_info.mode = 0o777
            asset_info.mtime = now
            asset_info.size = len(asset_bytes)
            tf.addfile(asset_info, io.BytesIO(asset_bytes))

pkg_size = os.path.getsize(output_package)
print(f"Successfully generated {output_package} (size: {pkg_size:,} bytes)")
