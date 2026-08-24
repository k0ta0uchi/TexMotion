# ✨ TexMotion

[English](#english) | [日本語](#日本語)

---

<a name="english"></a>
# English

**TexMotion** is an AI-powered Text-to-Motion generation extension for Unity and VRChat avatars. By integrating native in-engine inference via [kimodo.cpp](https://github.com/localai-org/kimodo.cpp), TexMotion enables avatar creators to generate fluid, natural Humanoid 3D animations directly from natural language prompts, preview them in real-time within an interactive 3D viewport, and automatically bind them to VRChat avatars—either non-destructively using **Modular Avatar** or directly into **VRChat Expressions Menus & Action Layers**.

---

## 🌟 Key Features

* **⚡ Native In-Engine AI Inference**: Runs lightweight C++ GGUF diffusion inference (`kimodo.dll` / `ggml-vulkan`) locally inside the Unity Editor with zero external Python runtime required.
* **🎬 Interactive 3D Motion Viewport**: Real-time 60fps 3D preview powered by `PreviewRenderUtility`, featuring orbit camera controls, zoom, timeline scrub, play/pause, and reset.
* **🦴 Native Humanoid Muscle Curve Baking**: Converts SMPL-X22 diffusion motion data into Unity native 95-Muscle + RootT/Q curves via `HumanPoseHandler` for 100% universal compatibility across all avatar shapes and sizes.
* **🖐️ Hand Pose & Facial Emotion Assistance**:
  * **Hand Presets**: `Natural Relaxed`, `Fist`, `Open Palm`, `Peace`, `Point`, or `Keep Free`.
  * **Facial Emotion AI**: Automatically detects keywords (e.g. *smile*, *angry*, *wink*, *surprised*) from prompts and binds avatar blendshapes with customizable intensity.
* **🦊 Dual VRChat Setup Modes**:
  * **Modular Avatar (Non-Destructive)**: Automatically generates merge animators and menu items. If Modular Avatar is missing, TexMotion offers one-click automated UPM installation!
  * **Direct VRCSDK**: Directly writes into `VRCAvatarDescriptor`, `VRCExpressionsMenu`, and `VRCExpressionParameters` with automatic Action Layer weight controls and intelligent **NextPage** pagination for full menus (8/8 slots).
* **📚 Motion Library Manager**: Scan, preview, apply, or cleanly remove generated motions with full reversibility.
* **⬇️ One-Click Hugging Face Downloader**: Automatically downloads motion diffusion GGUF models and text bundles from Hugging Face directly within the Editor.

---

## 🚀 Getting Started

### Requirements
* **Unity**: `2022.3.x` (or newer)
* **VRCSDK**: VRCSDK3 Avatar (`com.vrchat.avatars` 3.5.0+)
* *(Optional)* **Modular Avatar**: 1.9.0+

### Installation via VCC / Git URL (Package Manager)
1. In Unity, open **Window > Package Manager**.
2. Click the `+` button in the top-left and select **Add package from git URL...**.
3. Enter:
   ```text
   https://github.com/k0ta0uchi/TexMotion.git
   ```
4. Or import the `.unitypackage` from the [Latest Release](https://github.com/k0ta0uchi/TexMotion/releases).

---

## 📖 How to Use

1. Open **Tools > TexMotion > Motion Studio** from the Unity top menu.
2. Select your VRChat Avatar in the Hierarchy.
3. If models are missing, switch to **Settings & Models** and click **📥 Download Models from Hugging Face**.
4. Enter your motion prompt (e.g., `"a person throws a sharp right punch forward with energy"`).
5. Click **🚀 1. Generate Motion Preview** to inspect the motion in the interactive 3D viewport.
6. Click **✨ Apply to Avatar** to instantly bind the emote to your avatar!

---

<a name="日本語"></a>
# 日本語

**TexMotion** は、自然言語テキストから高品質な 3D モーションを生成し、Unity および VRChat アバターへ即座に組み込むことができる AI モーション作成拡張機能です。[kimodo.cpp](https://github.com/localai-org/kimodo.cpp) のネイティブ推論エンジンを Unity 内に統合しており、Python 環境の構築不要でローカル高速推論を実現します。

---

## 🌟 主な機能

* **⚡ Unity 内蔵ネイティブ高速推論**: `kimodo.dll` および `ggml-vulkan` による高速 GGUF 拡散推論を Unity エディタ内で直接実行。外部 Python 環境は一切不要です。
* **🎬 インタラクティブ 3D プレビュービューポート**: ウィンドウ内でアバターのクローンがリアルタイムに動作。ドラッグによるカメラ 360 度回転、ホイールズーム、タイムラインシークバー、Play/Pause を完備。
* **🦴 完全な Humanoid Muscle（筋肉値）ベイク**: Unity 公式の `HumanPoseHandler` を採用し、SMPL-X22 の骨格データを 95 本の Muscle カーブ＋RootT/Q に変換。身長や体型を問わずあらゆる Humanoid アバターで 100% 確実に動作します。
* **🖐️ ハンドポーズ & 表情アシスト機能**:
  * **指ポーズプリセット**: `Natural Relaxed (自然な手)`、`Fist (グー)`、`Open Palm (パー)`、`Peace (ピース)`、`Point (指差し)`、`Keep Free (ハンドサイン連動)`。
  * **表情 AI 自動推論**: プロンプト内のキーワード（*smile*, *angry*, *wink*, *surprise* 等）から感情を推論し、アバターの BlendShape（MMD名/英語名）と自動連動。
* **🦊 選べる 2 つの VRChat セットアップ方式**:
  * **Modular Avatar 連携（非破壊）**: アバター直下にマージ用オブジェクトを自動生成。未導入時はワンクリックで UPM 自動インストールをサポート。
  * **Direct VRCSDK（直接登録）**: `VRCAvatarDescriptor`、`VRCExpressionsMenu`、`VRCExpressionParameters`、および Action Layer へ直接登録。メニューが満杯（8/8スロット）の場合はスマートな **NextPage ページネーション**（退避・復元・連鎖的折りたたみ）を自動実行。
* **📚 モーションライブラリ管理**: 過去に生成したモーションの一覧表示、再プレビュー、アバターからの安全な一括削除・着脱。
* **⬇️ Hugging Face ワンクリックダウンローダー**: 必要なモデルデータ（GGUF および Text Bundle）をエディタ内から進捗バー付きで自動ダウンロード。

---

## 🚀 導入手順

### 必要環境
* **Unity**: `2022.3.x` 以降
* **VRCSDK**: VRCSDK3 Avatar (`com.vrchat.avatars` 3.5.0 以降)
* *(推奨)* **Modular Avatar**: 1.9.0 以降

### インストール方法（Git URL）
1. Unity メニューの **Window > Package Manager** を開きます。
2. 左上の `+` ボタンから **Add package from git URL...** を選択します。
3. 以下の URL を入力して追加します：
   ```text
   https://github.com/k0ta0uchi/TexMotion.git
   ```
4. または [Releases](https://github.com/k0ta0uchi/TexMotion/releases) から最新の `.unitypackage` をダウンロードしてインポートします。

---

## 📖 使い方

1. Unity 上部メニューの **`Tools > TexMotion > Motion Studio`** を開きます。
2. ヒエラルキー上の VRChat アバターを選択します。
3. モデルが未ダウンロードの場合は **Settings & Models** タブから **「📥 Download Models from Hugging Face」** を実行します。
4. プロンプト（例: `a dynamic hip hop dance routine`）を入力します。
5. **「🚀 1. Generate Motion Preview」** をクリックして 3D プレビューで動きを確認します。
6. **「✨ Apply to Avatar」** をクリックするだけで、アバターのエクスプレッションメニューへ即座にエモートが登録されます！

---

## 📄 License
This project is licensed under the MIT License.
Native GGUF diffusion inference powered by [kimodo.cpp](https://github.com/localai-org/kimodo.cpp).
