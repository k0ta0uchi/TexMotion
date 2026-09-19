# TexMotion

[English](#english) | [日本語](#日本語)

---

<a name="english"></a>
# English

**TexMotion** is an AI-powered motion generation extension for Unity and VRChat avatars. It generates 3D humanoid animations from natural language text prompts or extracts poses from 2D video files, and automatically configures them on VRChat avatars—either non-destructively through **Modular Avatar** or directly into **VRChat Expressions Menus and Action Layers**.

By integrating the native C++/GGML inference engine via [kimodo.cpp](https://github.com/localai-org/kimodo.cpp), TexMotion executes diffusion inference locally within the Unity Editor without requiring an external Python environment for text-driven generation.

<p align="center">
  <img src="docs/images/timeline_editor.png" alt="TexMotion Motion Timeline Editor" width="100%" />
</p>

---

## Key Features

* **In-Engine Local Inference**: Executes GGUF diffusion inference directly inside the Unity Editor using `kimodo.dll` and `ggml-vulkan`. No external Python environment is required for text-to-motion generation.
* **Video-to-Motion Pose Extraction**: Extracts humanoid 3D motions from common video formats (`.mp4`, `.mov`, etc.). Combines MediaPipe keypoint detection with WHAM global 3D pose lifting to mitigate tracking degradation caused by limb crossing or occlusion.
* **Motion Timeline and Pose Editor**: Provides a dedicated visual timeline editor to inspect and adjust animation clips frame-by-frame. Supports per-joint Euler angle and root position editing, pose copying and mirroring, jitter reduction via Slerp interpolation, and variable playback speeds (0.1x to 2.0x).
* **Advanced Pose Correction Suite**: Features 11 specialized repair tools and a 9-filter **Stylized Hand-Keyed Polish** engine: 3D onion skinning, 2-Bone IK dragging, In/Out range masking, easing tweens, loop boundary blending, additive range offsets, armpit limiters, foot grounding/sliding locks, glitch auto-repair, 8-slot pose palette, range retiming, plus hand-keyed timing filters (snap & ease, moving holds, anime 2s/3s stepping, landing cushion & bounce, contrapposto booster, kinematic chain delay, deceleration overshoot, dynamic pose exaggeration, and 3D trajectory arc smoothing). (See the [Advanced Pose Correction & Stylized Polish Guide](docs/timeline_editor_advanced_guide.md) for detailed tutorials and specifications).
* **Interactive 3D Preview Viewport**: Renders real-time previews using `PreviewRenderUtility`. Automatically adapts render target dimensions to vertical or horizontal video aspect ratios, and provides camera orbit, zoom controls, and a synchronized side-by-side video overlay.
* **Humanoid Muscle Curve Baking**: Converts estimated skeletal motions into Unity native 95-Muscle values and Root translation/rotation curves via `HumanPoseHandler`. The baked animations remain compatible across avatars of differing proportions and body scales.
* **Hand Pose and Facial Expression Assistance**:
  * **Hand Pose Presets**: Offers presets for natural relaxed hands, fists, open palms, peace signs, pointing, or preserving default controller hand signs.
  * **Facial Emotion Association**: Detects emotion keywords (such as smile, angry, wink, or surprised) within prompts and binds corresponding avatar blendshapes into the animation clip.
* **Dual VRChat Avatar Setup Modes**:
  * **Modular Avatar Setup (Non-Destructive)**: Generates merge animators and menu objects under the avatar hierarchy. If Modular Avatar is not present in the project, an automated package installation prompt is provided.
  * **Direct VRCSDK Setup**: Writes directly into `VRCAvatarDescriptor`, `VRCExpressionsMenu`, `VRCExpressionParameters`, and Action Layer animator controllers. If a target menu has reached its 8-item capacity, submenu pagination is created automatically.
* **Categorized Settings Interface**: Groups configuration options into General, Motion Generation, and Video Motion tabs to streamline workflow and model asset management.
* **Motion Library Manager**: Scans and lists generated clips (`Assets/TexMotion/Generated/`), allowing users to preview, apply, or safely remove configurations from avatars.
* **Model Asset Downloader**: Downloads required GGUF diffusion models, text embedding bundles, and video pose estimation weights from Hugging Face directly within the Editor.

---

## System Requirements

* **Unity**: `2022.3.x` or newer
* **VRCSDK**: VRCSDK3 - Avatars (`com.vrchat.avatars` 3.5.0 or newer)
* **Modular Avatar** (Optional): `1.9.0` or newer

---

## Installation

### Package Manager via Git URL

1. In Unity, open **Window > Package Manager**.
2. Click the **+** button in the top-left corner and select **Add package from git URL...**.
3. Enter the following repository URL:
   ```text
   https://github.com/k0ta0uchi/TexMotion.git
   ```

### UnityPackage

Download the latest `TexMotion.unitypackage` from [Releases](https://github.com/k0ta0uchi/TexMotion/releases) and import it into your project.

---

## Getting Started

1. Open **Tools > TexMotion > Motion Studio** from the Unity top menu.
2. Select your VRChat avatar in the Hierarchy.
3. If model assets are missing, open the **Settings & Models** tab and click **Download Models from Hugging Face**.
4. Enter a motion prompt (e.g., `"a person throws a sharp right punch forward with energy"`).
5. Click **1. Generate Motion Preview** to inspect the animation in the 3D viewport.
6. Click **Apply to Avatar** to configure the emote onto the selected avatar.

---

## Video-to-Motion Model Configuration

Video pose extraction supports multiple model backends depending on local compute resources and precision requirements.

### Model Storage and Acquisition

In **Settings & Models > Video2Motion Model Assets**, configure the video model storage directory (default: `%APPDATA%/TexMotion/Models/Video`). Checkpoints for RTMPose variants, DWPose WholeBody, MediaPipe Pose Landmarker task files, and optional WHAM, HMR2, or HybrIK weights can be downloaded individually or in bulk via **Download All**. Each row includes a link to its upstream repository for offline provisioning.

### WHAM Backend Architecture and Limitations

Downloading the WHAM checkpoint automatically deploys the bundled adapter script (`texmotion_wham_adapter.py`). When WHAM is selected under the **Pose Estimation Backend** dropdown, TexMotion executes its native WHAM temporal core (the official MotionEncoder fed with local MediaPipe 2D inputs) to perform 17-joint 3D lifting without requiring an external WHAM repository.

The official `wham_vit_w_3dpw.pth.tar` file contains neural network weights (state-dict) only; it does not include ViTPose or DPVO preprocessing pipelines, image-feature integrators, SMPL decoders, or licensed SMPL body models. To run a complete external pipeline including these stages, set the path to your external implementation via the `TEXMOTION_WHAM_ADAPTER_IMPL` environment variable. If the native core fails to load the checkpoint or its auxiliary detector, execution falls back to MediaPipe, logging the specific cause in the results card and Timeline Editor.

### Companion Assets and Data Dependencies

The Full-Parity Companion Assets section provides downloads for ViTPose-Huge, DPVO weights, and a baseline camera configuration template. Running ViTPose requires a local feature extractor or a precomputed `(frames, feature_dim)` archive. DPVO requires a local runtime capable of producing per-frame camera poses. The **Install Camera Template** action generates a template file (`camera.yaml`), which must be modified to match the optical characteristics of the source video. Licensed SMPL/SMPL-X body models cannot be distributed automatically; users must provide these files manually and set their file paths in settings.

### HMR2 Backend

When using HMR2, the bundled adapter `pose_pipeline/adapters/texmotion_hmr2_adapter.py` is invoked automatically. The upstream 4D-Humans runtime, its model archive (`hmr2a_model.tar.gz` or `hmr2_data.tar.gz`), and the neutral SMPL model (`.pkl`) must be installed manually. Configure these paths in Settings and run **Runtime Preflight** to verify the environment prior to extraction. If dependencies are missing or initialization fails, the error details are displayed before falling back to MediaPipe.

### Upstream References

* [RTMPose ONNX](https://huggingface.co/bukuroo/RTMPose-ONNX)
* [DWPose ONNX](https://huggingface.co/SceneWorks/dwpose-onnx)
* [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
* [WHAM](https://github.com/yohanshin/WHAM)
* [HMR2 / 4D-Humans](https://github.com/shubham-goel/4D-Humans)
* [HybrIK](https://github.com/jeffffffli/HybrIK)

---

<a name="日本語"></a>
# 日本語

**TexMotion** は、テキストプロンプトまたは 2D 動画から Unity Humanoid 形式の 3D アニメーションを生成・抽出し、VRChat 向けアバターへ組み込むための Unity エディタ拡張機能です。[kimodo.cpp](https://github.com/localai-org/kimodo.cpp) のネイティブ推論エンジンを Unity 内部に統合しており、外部の Python 実行環境を構築することなく、エディタ内でローカルに推論を実行できます。

<p align="center">
  <img src="docs/images/timeline_editor.png" alt="TexMotion タイムラインエディター" width="100%" />
</p>

---

## 主な機能

* **Unity 内でのローカル推論**: `kimodo.dll` および `ggml-vulkan` による GGUF 形式モデルの拡散推論を Unity エディタ内で直接実行します。テキストからのモーション生成において、外部 Python 環境を用意する必要はありません。
* **動画からのモーション抽出**: 一般的な動画ファイル（`.mp4` や `.mov` など）から Humanoid 3D モーションを抽出します。MediaPipe によるキーポイント検出と WHAM によるグローバル 3D 姿勢推定を組み合わせることで、手足の交差や身体の重なりが生じる動作でもトラッキングの破綻を抑えます。
* **モーションタイムラインとポーズ編集**: 専用のタイムラインエディタを備えており、生成したクリップをフレーム単位で調整できます。関節ごとのオイラー角やルート移動の編集、ポーズのコピーと反転貼り付け、Slerp 補間によるブレの抑制、0.1 倍から 2.0 倍までの可変速再生に対応します。
* **高度ポーズ修正ツール群＆手付け風クオリティ向上**: 姿勢推定時のノイズ修正ツール 11 種に加え、プロの手付けアニメーション品質へ昇華する 9 種類の **手付け風クオリティ向上（Stylized Polish）** 機能を統合しています。タメ・ツメ強調（Snap & Ease）、アニメコマ打ち（2コマ／3コマ打ち）、キーフレーム削減、接地衝撃クッション＆バウンス、コントラポスト（骨盤傾斜強調）、四肢遅延しなり（Kinematic Drag）、停止時反動（Overshoot）、ポーズ誇張（Pose Exaggeration）、および 3D 円弧整流（Trajectory Arc Smoother）により、モキャプ特有の均一速度や浮遊感を劇的に改善します（詳細は [高度ポーズ修正＆手付け風クオリティ向上ガイド](docs/timeline_editor_advanced_guide.md) を参照）。
* **インタラクティブな 3D プレビュー**: `PreviewRenderUtility` を用いた専用ビューポートで動作を確認できます。縦型動画や横型動画のアスペクト比に合わせて描画サイズを自動調整し、カメラの視点回転、拡大縮小、元動画と検出点の重ね合わせ表示を行えます。
* **Humanoid マッスルカーブへの変換**: Unity 標準の `HumanPoseHandler` を介して、推論結果の骨格データを 95 系統の Muscle 値および Root 移動・回転カーブに変換します。体型や骨格比率の異なる Humanoid アバターに対してもモーションを適用できます。
* **手・表情の補助設定**:
  * **指ポーズのプリセット**: 自然な開き手、握り拳、平手、ピース、指差し、または既存アニメーションの維持を選択できます。
  * **表情の自動連動**: プロンプト内のキーワード（smile、angry、wink、surprised など）を検出し、アバターの対応する表情 BlendShape を自動でアニメーションに組み込みます。
* **VRChat 向けセットアップ**:
  * **Modular Avatar 連携**: アバターの階層下に非破壊マージ用オブジェクトを自動生成します。プロジェクト内に Modular Avatar が存在しない場合は、ダイアログから自動導入できます。
  * **Direct VRCSDK 設定**: `VRCAvatarDescriptor`、`VRCExpressionsMenu`、`VRCExpressionParameters`、および Action レイヤーのアニメーターコントローラへ直接書き込みます。登録先メニューが上限の 8 項目に達している場合は、自動的にサブメニューを階層化して退避と登録を行います。
* **整理された設定画面**: 設定項目を「一般」「モーション生成」「動画モーション」の 3 つのカテゴリに整理し、目的に応じた設定やモデル管理を行えます。
* **モーションライブラリ管理**: 生成したアニメーションクリップ（`Assets/TexMotion/Generated/`）の一覧表示、再プレビュー、アバターへの適用および安全な削除を行えます。
* **モデルダウンローダー**: 動作に必要な GGUF モデルやテキスト埋め込みモデル、動画解析用重みデータを、エディタ内の設定画面から直接ダウンロードできます。

---

## 動作環境

* **Unity**: `2022.3.x` 以降
* **VRCSDK**: VRCSDK3 - Avatars (`com.vrchat.avatars` 3.5.0 以降)
* **Modular Avatar**（任意）: `1.9.0` 以降

---

## インストール手順

### Package Manager による導入（Git URL）

1. Unity エディタのメニューから **Window > Package Manager** を開きます。
2. ウィンドウ左上の **+** ボタンを押し、**Add package from git URL...** を選択します。
3. 以下の URL を入力してパッケージを追加します：
   ```text
   https://github.com/k0ta0uchi/TexMotion.git
   ```

### UnityPackage による導入

[Releases](https://github.com/k0ta0uchi/TexMotion/releases) から最新の `TexMotion.unitypackage` をダウンロードし、対象プロジェクトへインポートします。

---

## 使用手順

1. Unity のメニューから **Tools > TexMotion > Motion Studio** を開きます。
2. Hierarchy で対象の VRChat アバターを選択します。
3. モデルが未ダウンロードの場合は、**Settings & Models** タブで **Download Models from Hugging Face** を実行します。
4. プロンプト（例: `a person throws a sharp right punch forward with energy`）を入力します。
5. **1. Generate Motion Preview** を押して、3D ビューポートでモーションを確認します。
6. **Apply to Avatar** を押すと、選択中のアバターにモーションが登録されます。

---

## 動画モーション抽出（Video2Motion）の構成とモデル

動画からのモーション抽出では、利用するモデルや処理方式を環境に応じて切り替えられます。

### モデルの保存先と入手

**Settings & Models > Video2Motion Model Assets** でモデルの保存ディレクトリ（既定値: `%APPDATA%/TexMotion/Models/Video`）を指定します。RTMPose、DWPose WholeBody、MediaPipe Pose Landmarker のタスクファイル、および任意の WHAM、HMR2、HybrIK チェックポイントを個別または一括で取得できます。各項目には配布元のリンクが記載されており、手動でのファイル配置にも対応しています。

### WHAM バックエンドの仕様と制約

WHAM チェックポイントをダウンロードすると、同梱のアダプタースクリプト（`texmotion_wham_adapter.py`）が自動で配置されます。Video Motion タブの **Pose Estimation Backend** で WHAM を選択すると、内蔵の WHAM 時系列コア（公式 MotionEncoder と MediaPipe による 2D 入力）が実行され、外部リポジトリを導入することなく 17 関節の 3D 姿勢リフティングを行えます。

公式の配布ファイル `wham_vit_w_3dpw.pth.tar` はネットワークの重み（state-dict）のみであり、ViTPose や DPVO による前処理、画像特徴の統合器、SMPL デコーダ、SMPL のモデルデータ自体は含まれません。これらを含む完全な外部パイプラインを実行する場合は、環境変数 `TEXMOTION_WHAM_ADAPTER_IMPL` で外部実装のパスを指定します。内蔵コアでチェックポイントや補助検出器の読み込みに失敗した場合は、自動的に MediaPipe による推定へ切り替わり、その理由が結果画面およびタイムラインエディタに記録されます。

### 追加アセットの取り扱い

Settings 画面の Full-Parity Companion Assets では、ViTPose-Huge や DPVO のチェックポイント、およびカメラ設定テンプレートを取得できます。ViTPose の利用にはローカルの特徴抽出環境またはフレーム特徴量アーカイブが必要であり、DPVO の利用には各フレームのカメラ姿勢を出力する実行環境が必要です。カメラ設定テンプレート（`camera.yaml`）は初期設定ファイルを出力しますが、解析対象の動画に合わせた内部パラメータの入力が必要です。ライセンスの必要な SMPL/SMPL-X の身体モデルデータは自動ダウンロードの対象外であり、利用者が各自で用意してファイルパスを指定します。

### HMR2 バックエンド

HMR2 を利用する場合は、同梱の `pose_pipeline/adapters/texmotion_hmr2_adapter.py` が自動で使用されます。公式の 4D-Humans 実行環境、チェックポイント（`hmr2a_model.tar.gz` または `hmr2_data.tar.gz`）、および中立 SMPL モデル（`.pkl`）をあらかじめ準備する必要があります。PyTorch 環境にランタイムをインストールしたうえで、Settings 画面で各ファイルパスを指定し、**Runtime Preflight** で検証を行ってから抽出を実行します。必要なファイルが不足している場合や初期化に失敗した場合は、エラー内容を表示したうえで MediaPipe へのフォールバックが行われます。

### 参照リソース

* [RTMPose ONNX](https://huggingface.co/bukuroo/RTMPose-ONNX)
* [DWPose ONNX](https://huggingface.co/SceneWorks/dwpose-onnx)
* [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
* [WHAM](https://github.com/yohanshin/WHAM)
* [HMR2 / 4D-Humans](https://github.com/shubham-goel/4D-Humans)
* [HybrIK](https://github.com/jeffffffli/HybrIK)

---

## ライセンス

本プロジェクトは MIT License のもとで公開されています。
拡散モデルのネイティブ推論エンジンには [kimodo.cpp](https://github.com/localai-org/kimodo.cpp) を使用しています。
