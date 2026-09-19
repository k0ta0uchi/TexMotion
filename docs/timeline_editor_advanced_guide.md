# タイムラインエディタ 高度ポーズ修正ガイド / Advanced Pose Correction Guide

本ドキュメントでは、TexMotion のタイムラインエディタ（`MotionTimelineEditorWindow`）に搭載された高度ポーズ修正機能の動作原理、操作手順、および技術仕様を解説します。

---

## 目次 / Table of Contents

- [日本語ガイド](#日本語ガイド)
  - [1. 概要](#1-概要)
  - [2. ビューポート支援ツール & HUD ツールバー](#2-ビューポート支援ツール--hud-ツールバー)
    - [2.1 3D オニオンスキン表示](#21-3d-オニオンスキン表示)
    - [2.2 3D ビューポート 2-Bone IK 操作](#22-3d-ビューポート-2-bone-ik-操作)
    - [2.3 3D ステージ・背景切替ドロップダウン](#23-3d-ステージ背景切替ドロップダウン)
    - [2.4 自動接地プレビュー切替（接地トグル）](#24-自動接地プレビュー切替接地トグル)
  - [3. 4大タブ構成インスペクター概要](#3-4大タブ構成インスペクター概要)
  - [4. 【タブ1: ポーズ (Pose)】部位マスク・ポーズアクション・パレット](#4-タブ1-ポーズ-pose部位マスクポーズアクションパレット)
    - [4.1 部位別マスク (Body Part Masking)](#41-部位別マスク-body-part-masking)
    - [4.2 ポーズアクション (Pose Actions)](#42-ポーズアクション-pose-actions)
    - [4.3 8スロット ポーズパレットとブレンド適用](#43-8スロット-ポーズパレットとブレンド適用)
  - [5. 【タブ2: 修復 (Repair)】幾何制約・接触・ジッター修正](#5-タブ2-修復-repair幾何制約接触ジッター修正)
    - [5.1 範囲加算オフセット (Additive Range Offset)](#51-範囲加算オフセット-additive-range-offset)
    - [5.2 脇部・胸部めり込み防止リミッター](#52-脇部胸部めり込み防止リミッター)
    - [5.3 地面接地スナップと足位置固定 (Ground Snap & Lock Feet)](#53-地面接地スナップと足位置固定-ground-snap--lock-feet)
    - [5.4 異常フレーム・ジッター自動検出と一括修復 (Glitch Auto-Fixer)](#54-異常フレームジッター自動検出と一括修復-glitch-auto-fixer)
  - [6. 【タブ3: タイミング (Timing)】時間軸・キーフレーム編集](#6-タブ3-タイミング-timing時間軸キーフレーム編集)
    - [6.1 In / Out 区間指定](#61-in--out-区間指定)
    - [6.2 範囲Tween補間（キーフレーム・トゥイーン）](#62-範囲tween補間キーフレームトゥイーン)
    - [6.3 リタイミング（区間の伸縮・Time Warp）](#63-リタイミング区間の伸縮time-warp)
    - [6.4 ループ境界ブレンダー (Seamless Loop Blender)](#64-ループ境界ブレンダー-seamless-loop-blender)
  - [7. 【タブ4: ポリッシュ (Polish)】手付け風クオリティ向上](#7-タブ4-ポリッシュ-polish手付け風クオリティ向上)
    - [7.1 推奨プリセット](#71-推奨プリセット)
    - [7.2 9つの機能と技術仕様](#72-9つの機能と技術仕様)
    - [7.3 操作手順](#73-操作手順)
- [English Guide](#english-guide)
  - [1. Overview](#1-overview)
  - [2. Viewport Assistance Tools & HUD](#2-viewport-assistance-tools--hud)
  - [3. 4-Tab Inspector Overview](#3-4-tab-inspector-overview)
  - [4. [Tab 1: Pose] Masking, Actions & Palette](#4-tab-1-pose-masking-actions--palette)
  - [5. [Tab 2: Repair] Geometric Constraints, Grounding & Glitch Fix](#5-tab-2-repair-geometric-constraints-grounding--glitch-fix)
  - [6. [Tab 3: Timing] Temporal & Keyframe Editing](#6-tab-3-timing-temporal--keyframe-editing)
  - [7. [Tab 4: Polish] Stylized Hand-Keyed Polish](#7-tab-4-polish-stylized-hand-keyed-polish)

---

# 日本語ガイド

## 1. 概要

動画からの姿勢推定処理（HMR2 / WHAM / ViTPose）では、被写体の遮蔽、高速移動によるブレ、または視覚的なあいまいさに起因して、関節角度の急激な跳躍や四肢のめり込み、足の浮き上がりが発生することがあります。TexMotion は、推定された 3D モーションデータを Unity 上で直感的に調整するための専用タイムラインエディタを備えています。

本エディタは操作性とワークフローが大幅にリファクタリングされ、ビューポート支援 HUD ツールバーに加え、右側インスペクターが **「ポーズ」「修復」「タイミング」「ポリッシュ」** の 4 大タブ構成に統合・整理されています。

![タイムラインエディタ全体図](images/timeline_editor_overview.jpg)

---

## 2. ビューポート支援ツール & HUD ツールバー

ビューポート上部の HUD（Heads-Up Display）ツールバーには、3D アバターの視認性と操作性を高める各種トグルおよび設定が集約されています。

### 2.1 3D オニオンスキン表示
手動修正を行う際、前後フレームとの連続性を視覚的に確認できない状態では、滑らかな軌道を作り出すことが困難になります。オニオンスキン機能は、現在のフレームの前後に位置する姿勢を半透明のゴーストメッシュとしてビューポート上に重畳描画します。

- **表示仕様**:
  - 直前フレーム（Frame - 1）: シアン色（水色）の半透明シルエットで描画されます。
  - 直後フレーム（Frame + 1）: マゼンタ色（赤紫色）の半透明シルエットで描画されます。
- **操作手順**:
  1. HUD ツールバーの「🧅 Onion」ボタンを押下して有効化します。
  2. タイムラインスライダーを移動させると、前後の姿勢差が色分けされて表示されます。

### 2.2 3D ビューポート 2-Bone IK 操作
腕や脚の末端（手首・足首）を目標位置へ移動させたい場合、肩・肘・手首の3関節を個別に回転させる手法は調整コストが高くなります。本機能は、ビューポート上に操作用ピンを提示し、ドラッグ操作によって解析的 2-Bone IK（2関節逆運動学）を計算して関節角度へ反映します。

- **対象関節**:
  - 左手首（`SmplxJoint.L_Wrist`）、右手首（`SmplxJoint.R_Wrist`）
  - 左足首（`SmplxJoint.L_Ankle`）、右足首（`SmplxJoint.R_Ankle`）
- **アルゴリズム**:
  - 余弦定理（Law of Cosines）を用いた解析的解法により、目標点への到達に必要な親関節と中間関節の回転量を瞬時に算出します。
  - 四肢の最大長を超える位置へドラッグされた場合は、到達限界距離へ自動的にクランプされます。
- **操作手順**:
  1. HUD ツールバーの「🦾 IKピン」ボタンを押下します。
  2. 手首または足首の位置に黄色の操作ピンが表示されます。
  3. ピンを左ドラッグすると、四肢がリアルタイムに追従して変形します。マウスボタンを離した時点で編集が確定し、Undo 履歴へ記録されます。

### 2.3 3D ステージ・背景切替ドロップダウン
用途に応じて 3D プレビューの背景環境をドロップダウンメニューから瞬時に切り替えることができます。
- **ステージ (Stage)**: チェッカーグリッド床面、動的コンタクトシャドウ、深度Z遮蔽を備えたスタジオ空間。足の設置感や空間的な距離関係を直感的に把握できます。
- **ダーク (Dark)**: 従来の漆黒背景。骨格ギズモやオニオンスキンメッシュの高コントラストな確認に最適です。
- **グリーン (Green)**: クロマキー合成用の高純度グリーンバック。OBS や動画編集ソフトでの切り抜き素材収録に活用できます。

### 2.4 自動接地プレビュー切替（接地トグル）
AI 推定直後のモーションデータにおいて、足先が地面から浮いて見えたり、逆に沈み込んでいるように見える場合があります。HUD の「接地」ボタンをオンにすると、最下点の足高さを自動検出し、床面 $Y=0$ に自然に接地した状態でプレビュー表示されます。

---

## 3. 4大タブ構成インスペクター概要

タイムラインエディタ右側のインスペクターパネルは、作業目的別に 4 つの独立したタブモジュールに分類されています。

![機能詳細図](images/timeline_editor_tools_detail.jpg)

| タブ | 名称 | 主な機能と対象 |
| :--- | :--- | :--- |
| **1. ポーズ** | Pose & Action | 関節個別回転、部位別マスク、ポーズコピー/貼付/反転、8スロットポーズパレット |
| **2. 修復** | Repair & Contact | 範囲加算オフセット（腰高さ・腕開き角）、脇めり込みリミッター、接地スナップ＆足固定、グリッチ自動検出・一括修復 |
| **3. タイミング** | Timing & Tween | In/Out 区間指定、範囲Tween補間（5種イージング）、リタイミング（区間伸縮）、ループ境界ブレンダー |
| **4. ポリッシュ** | Stylized Polish | 手付け風モーションポリッシュ（Action/Weight/Subtle プリセット、9大数理運動学フィルター） |

---

## 4. 【タブ1: ポーズ (Pose)】部位マスク・ポーズアクション・パレット

### 4.1 部位別マスク (Body Part Masking)
全身のモーションを変更することなく、例えば「下半身のステップは維持したまま、上半身の向きだけを修正する」「左腕の動きだけをコピー＆ペーストする」といった作業を可能にするビットマスク機構です。

- **選択可能なプリセット**:
  - `All`: 全身の 22 関節およびルート位置
  - `UpperBody`: 脊椎（Spine1〜3）、頭部（Neck, Head）、両腕
  - `LowerBody`: 骨盤（Pelvis / Root）、両脚
  - `Arms`: 両腕（鎖骨、肩、肘、手首）
  - `Legs`: 両脚（股関節、膝、足首、足先）

### 4.2 ポーズアクション (Pose Actions)
現在のフレームまたはクリップボードを対象に、頻出のポーズ編集をワンクリックで実行します。
- **ポーズをコピー / ポーズを貼り付け**: アクティブな部位マスクに基づいて、選択部位のみの姿勢をクリップボード経由で他フレームへ適用します。
- **ポーズを反転 (Mirror)**: 左右対称なポーズ反転を行います（歩行の逆足ポーズ作成などに便利です）。
- **フレームをスムーズ**: 前後フレームと Slerp ブレンドを行い、特定フレームの突発的な揺れを平滑化します。
- **フレームをリセット**: 元の推定姿勢へ復元します。
- **Tポーズ**: 基準となる標準 T ポーズへリセットします。

### 4.3 8スロット ポーズパレットとブレンド適用
頻繁に使用する姿勢（綺麗なニュートラルポーズ、固有の構え姿勢など）を 8 つのスロットに一時保存し、任意のフレームへ任意の割合でブレンド適用できます。

- **操作手順**:
  1. スロット番号（1〜8）を選択します。
  2. 「💾 Store to Slot」を押下して現在の姿勢を記録します（登録済みスロットは緑色 ● で示されます）。
  3. 修正したいフレームへ移動し、「Blend Weight」スライダー（0.0〜1.0）を設定します。
  4. 部位マスクを選択した状態で「Apply Blend」を押下すると、部分的な姿勢反映が実行されます。

---

## 5. 【タブ2: 修復 (Repair)】幾何制約・接触・ジッター修正

### 5.1 範囲加算オフセット (Additive Range Offset)
指定した In-Out 範囲の全フレームに対して、一定の姿勢変形を加算します。
- **腰の高さオフセット (Hips Y)**: キャラクタ全体の腰高（浮遊感の解消や屈み具合）をメートル単位で上下させます。
- **腕の開き角 (Arm Open Angle)**: 両肩の開き角（Tポーズ方向 / 閉じ方向）を一括調整します。
- **範囲の境界をフェード (Fade at Range Edges)**: 有効にすると、区間の両端において重みを自動的に 0 へエルミート減衰させ、区間外のモーションとの不連続な段差を防ぎます。

### 5.2 脇部・胸部めり込み防止リミッター
タイトな服や体型モデルにおいて、推定された腕が胸部や脇腹の内側へ突き抜けてしまう問題（ペネトレーション）を抑制します。左右の肩関節のローカル角度を検査し、体幹に近づきすぎている場合に、指定した下限角度へと安全にクランプします。

### 5.3 地面接地スナップと足位置固定 (Ground Snap & Lock Feet)
歩行や足踏みにおいて、足先が地面（床面）を突き抜けてしまう現象や、接地中に足が前後左右へ滑る現象（フットスライディング）を補正します。

- **Ground Snap**:
  - SMPL-X 骨格の順運動学（FK）により、両足首および足先のワールド高さを評価します。
  - 床面高さ（`Ground Plane Y`）を下回ったフレームを検出し、接地に必要な量だけルート位置（骨盤）の高さを自動補正します。
- **Lock Feet**:
  - 接触開始フレーム（In点）における足首の 3D 空間位置を固定目標とし、区間内の各フレームに対して足の 2-Bone IK を解いて足裏を一定位置に留めます。

### 5.4 異常フレーム・ジッター自動検出と一括修復 (Glitch Auto-Fixer)
トラッキング外れによって発生する、1フレームだけ関節が激しく跳ねる異常（グリッチ）を自動的に洗い出します。

- **検出基準**:
  1. **過大角速度**: 連続するフレーム間で関節の回転角速度が 720°/秒 を超過している。
  2. **急反転スパイク**: 関節が 110° 以上の急角度で跳躍し、直後のフレームで元の角度近傍（45°未満）へ急激に戻っている。
- **表示と修復**:
  - タイムライン上に赤色のバーが描画され、問題箇所のフレーム番号と原因関節が一覧表示されます。
  - 「Go」ボタンで該当フレームへ即座に移動できます。
  - 「⚡ Fix All Glitches」を押下すると、全検出フレームに対して近傍フレームからの球面線形補間（Slerp）が一括適用され、1回の操作で全修正が完了します。

---

## 6. 【タブ3: タイミング (Timing)】時間軸・キーフレーム編集

### 6.1 In / Out 区間指定
広範なモーションデータの中から特定の動作区間（歩行の1周期、跳躍の離陸から着地までなど）に限定して処理を適用するために、In / Out 範囲を設定します。設定された範囲は、タイムライン下部のルーラー帯に淡いアクア色の帯（IN / OUT マーカー付き）として可視化されます。

### 6.2 範囲Tween補間（キーフレーム・トゥイーン）
欠損したフレームや、極端なトラッキング破綻が発生した区間を、始点（In点）と終点（Out点）の姿勢に基づいて滑らかに補間します。

- **補間アルゴリズム**:
  - ルート移動（位置）: ベクトル線形補間（Lerp）によるスムーズな移動推移
  - 関節回転: 四元数球面線形補間（Quaternion.Slerp）
- **イージング曲線**:
  - `Linear`: 等速運動
  - `EaseIn`: 加速運動（二次曲線 $t^2$）
  - `EaseOut`: 減速運動（$1 - (1-t)^2$）
  - `EaseInOut`: 前後緩衝曲線
  - `SmoothStep`: エルミート補間曲線（$3t^2 - 2t^3$）
- **操作手順**:
  1. In点とOut点を設定します。
  2. ドロップダウンから希望のイージング型を選択します。
  3. 「Interpolate In ➔ Out (Tween)」ボタンを押下します。

### 6.3 リタイミング（区間の伸縮・Time Warp）
特定のアクション（パンチのタメ、スローモーション演出など）の再生速度を時間軸上で変更します。選択された In-Out 区間のフレーム数を、指定した目標フレーム数へとリサンプリングして伸縮します。等速 / 1.5倍 / 2倍 / 0.5倍 などのプリセットボタンで即座にテンポ調整が可能です。

### 6.4 ループ境界ブレンダー (Seamless Loop Blender)
待機モーションや走行アニメーションをゲームでループ再生する場合、最終フレームから先頭フレームへ遷移する瞬間に姿勢の不一致（ポップノイズ）が生じます。

- **機構**:
  - モーション末尾の指定フレーム数（Margin Frames）にわたり、先頭フレーム（Frame 0）の姿勢へと徐々に遷移するクロスフェード処理を行います。
  - ルート移動が InPlace モード（定位置再生）に設定されている場合は、水平位置（X, Z）の整合を保ちつつ、高さ（Y）を滑らかに接続します。
- **操作手順**:
  1. 「Margin Frames」でブレンドに費やすフレーム長（推奨: 6〜12フレーム）を指定します。
  2. 「Blend Loop Boundary」ボタンを押下します。

---

## 7. 【タブ4: ポリッシュ (Polish)】手付け風クオリティ向上（Stylized Hand-Keyed Polish）

単眼カメラからの AI 姿勢推定で得られたモーションは、全フレームにキーが記録されたベタ打ちデータとなるため、動作速度が均一で緩急に乏しく、全身が同時に動くロボット感や、浮遊感（ウェイト感の不足）が生じやすくなります。

TexMotion は、ディズニーの 12 原則や日本のアニメーション技法に基づく 9 種類の数理・運動学フィルターを備えており、均一なモキャプデータをプロの手付けアニメーションのようなキレと重力感のある動作へと一括昇華させます。

![手付け風クオリティ向上機能ガイド](images/timeline_editor_stylized_polish.jpg)

### 7.1 推奨プリセット
目的に応じてワンクリックで各パラメータを一括設定できる 3 種類のプリセットを用意しています。

| プリセット | 主な特徴とパラメータ傾向 | 推奨用途 |
| :--- | :--- | :--- |
| **⚡ Action (キレ重視)** | スナップ強度 0.70、ポーズ誇張 1.28x、接地クッション 4.5cm、可変コマ打ち（Dynamic Anime） | ゲームアクション、ダンス、ケレン味を重視する表現 |
| **🏋️ Weight (重量感重視)** | 接地クッション 4.0cm（5f復帰）、コントラポスト 1.30x、ドラッグしなり 0.9f、停止反動 0.30 | 歩行、走行、重量物の運搬、日常動作の接地感強化 |
| **🍃 Subtle (微調整・自然)** | スナップ強度 0.25、ポーズ誇張 1.08x、接地クッション 2.0cm、アーク整流 0.50 | 実写の自然さを損なわずに AI ノイズと浮遊感のみを除去 |

### 7.2 9つの機能と技術仕様

#### 1. タイミング＆スペーシング（キレと緩急）
- **Snap & Ease（タメ・ツメ強調）** `[初期設定: ON]`
  - 関節角速度が閾値（`Hold Threshold`、デフォルト 16°/s）以下の区間を「Moving Hold（微小呼吸のみ維持）」とし、高速遷移区間をエルミート S 字曲線（SmoothStep）でリスペーシングします。均一速度のヌメヌメ感を排除し、パッと動いてスッと止まるメリハリを形成します。
- **Anime Frame Stepping（可変コマ打ち）** `[初期設定: OFF]`
  - 日本のリミテッドアニメーション（作画アニメ）表現をエミュレートします。関節角速度に応じて動的に 1 コマ打ち（30fps: アクション）、2 コマ打ち（15fps: 通常動作）、3 コマ打ち（10fps: タメ・静止）のステップ保持を自動切り替えます（Strict 2s / 3s の固定モードも選択可能）。
- **Keyframe Decimator（キーフレーム削減）** `[初期設定: OFF]`
  - SO(3) 球面空間上の Ramer-Douglas-Peucker アルゴリズムにより、指定した許容角度誤差（`Tolerance`、デフォルト 2.5°）を満たす極値キーフレーム（Extremes / Breakdowns）のみを抽出し、中間フレームを削減してスプライン補間カーブへ再構成します。

#### 2. ウェイト＆フィジックス（重力と接地感）
- **Landing Cushion & Bounce（接地衝撃沈み込み）** `[初期設定: ON]`
  - SMPL-X 骨格の順運動学（FK）により足首および足先の垂直速度を計測し、足裏が地面に到達した着地（Touch-down）の瞬間を検知します。着地直後の骨盤（`RootPositions.y`）に下方向の沈み込み（`Cushion Depth`、デフォルト 3.5cm）を重畳し、減衰正弦波により指定フレーム（`Recovery Frames`、デフォルト 4f）で定位置へ復帰させます。
- **Contrapposto Booster（骨盤傾斜強調）** `[初期設定: ON]`
  - 左右の足の接地高さと荷重状態を判定し、体重が乗っている軸足側の骨盤（Pelvis roll）を上方へ傾斜させ、頭部と上半身のバランスを保つために胸郭（Spine1/Spine2）を逆方向へカウンターチルトさせます。人体の美しい立ちポーズと自然な体重移動を再現します。

#### 3. 運動連鎖としなり（ドラッグとフォロースルー）
- **Kinematic Drag（四肢遅延しなり）** `[初期設定: ON]`
  - 人体の階層構造（Pelvis → Spine → Shoulder/Neck → Elbow/Head → Wrist）に基づき、末端関節の回転位相を上流関節に対してサブフレーム単位（`Drag Delay`、デフォルト 0.85f）で遅延させます。全身が同時に動いて同時に止まるロボット感を排除し、鞭のような柔らかいしなりと余韻を創出します。
- **Overshoot & Settling（停止時反動）** `[初期設定: ON]`
  - 急速なブレーキ（角加速度の急減速）が発生したフレームにおいて、手首や頭部が目標ポーズを一時的に行き過ぎてから減衰振動で収束するスプリング慣性（`Overshoot Amount`、デフォルト 0.35）を自動付加します。

#### 4. ポーズとシルエットの洗練（誇張とアーク）
- **Pose Exaggeration（ポーズ誇張）** `[初期設定: ON]`
  - ニュートラル直立姿勢からの各関節の回転角変位をスケール増幅（`Exaggeration Scale`、デフォルト 1.18x）します。AI 推定が無難に小さく丸めてしまいがちなポーズを画面外側へ押し出し（Push）、シルエットの明瞭さと力強さを大幅に向上させます。
- **Trajectory Arc Smoother（3D円弧整流）** `[初期設定: ON]`
  - 手首および足首の 3D ワールド座標系列を時間軸平滑化し、カメラ推定ノイズによる直線的・ガタガタした軌道を、美しい円弧（アーク）へと整流します。整流後の目標位置に向けて解析的 2-Bone IK を解き、四肢の関節回転角を再計算します。

### 7.3 操作手順
1. タイムライン上で修正したい区間を [Set In] / [Set Out] で指定します（クリップ全編に適用する場合は「Entire Clip」を選択）。
2. 必要に応じて部位マスク（[Body Mask]）を選択し、修正部位を上半身や四肢に限定します。
3. プリセットボタン（[⚡ Action] / [🏋️ Weight] / [🍃 Subtle]）を押下し、目的に合致する設定を一括ロードします。
4. こだわりたいパラメータ（誇張スケールや沈み込み深さ等）があれば、各スライダーで微調整します。
5. 「✨ Polish Motion」ボタンを押下すると、全フィルターが順次適用され、ビューポート上の 3D アバターに即座に反映されます。
6. 「↩️ 元に戻す（Undo）」と「↪️ やり直す（Redo）」を用いて、修正前後の動作のキレや重力感を比較検証します。

---

# English Guide

## 1. Overview

Monocular motion capture and 3D pose estimation methods (such as HMR2, WHAM, and ViTPose) are prone to tracking noise, limb penetration, and abrupt joint flips caused by occlusions, high-velocity movements, or visual ambiguities. The TexMotion Timeline Editor (`MotionTimelineEditorWindow`) offers an integrated suite of advanced correction tools designed to inspect and refine 3D motion clips directly inside the Unity Editor.

The editor has been refactored into an intuitive workflow featuring an interactive viewport HUD toolbar and a streamlined **4-Tab Inspector** consisting of **Pose**, **Repair**, **Timing**, and **Polish**.

![Timeline Editor Overview](images/timeline_editor_overview.jpg)

---

## 2. Viewport Assistance Tools & HUD

The HUD toolbar located directly above the 3D viewport consolidates essential visualization and snapping controls.

### 2.1 3D Onion Skin Overlay
Provides visual continuity during manual keyframe editing by rendering translucent ghost silhouettes of adjacent frames directly in the 3D viewport.
- **Color Coding**:
  - Previous Frame ($t - 1$): Cyan translucent silhouette.
  - Next Frame ($t + 1$): Magenta translucent silhouette.
- **Usage**: Toggle via the `🧅 Onion` button in the viewport HUD.

### 2.2 Viewport 2-Bone IK Pin Manipulation
Enables intuitive dragging of limb extremities (wrists and ankles) instead of rotating individual hierarchical joints.
- **Target Joints**: Left/Right Wrists and Left/Right Ankles.
- **Algorithm**: Analytical Two-Bone Inverse Kinematics using the Law of Cosines. Clamps targets beyond reach to limb length limits.
- **Usage**: Toggle `🦾 IK Pins` in the HUD, then click and drag the yellow manipulator pins in the viewport. Rotations are automatically applied and recorded into the Undo stack upon mouse release.

### 2.3 3D Stage & Background Selector (Stage / Dark / Green)
Switch between rendering environments via the HUD dropdown:
- **Stage**: Studio floor with checkerboard grid, dynamic contact shadow, and Z-depth occlusion for accurate spatial grounding assessment.
- **Dark**: High-contrast pitch-black environment for inspecting skeleton gizmos and onion skin overlays.
- **Green**: Pure chroma-key green screen for recording video clips and streaming composite overlays.

### 2.4 Auto Foot Grounding Preview (Ground Toggle)
When active, automatically detects the lowest foot extremity point per frame and snaps the avatar vertically so that feet rest precisely on the ground plane ($Y=0$) during preview.

---

## 3. 4-Tab Inspector Overview

The inspector panel on the right organizes editing workflows into four dedicated modules.

![Advanced Pose Correction Tools Detail](images/timeline_editor_tools_detail.jpg)

| Tab | Name | Primary Capabilities |
| :--- | :--- | :--- |
| **1. Pose** | Pose & Action | Joint rotation, body part masking, copy/paste/mirror actions, 8-slot pose palette |
| **2. Repair** | Repair & Contact | Additive range offsets (hips & arms), armpit penetration limiter, ground snapping & foot locking, glitch auto-repair |
| **3. Timing** | Timing & Tween | In/Out range marking, 5-curve tween interpolation, time-warp retiming, seamless loop blender |
| **4. Polish** | Stylized Polish | Hand-keyed stylized animation polish (Action, Weight, and Subtle presets across 9 kinematic filters) |

---

## 4. [Tab 1: Pose] Masking, Actions & Palette

### 4.1 Body Part Masking
Allows selective editing across anatomical limb groupings without altering the rest of the skeleton.
- **Presets**: `All`, `UpperBody`, `LowerBody`, `Arms`, `Legs`.

### 4.2 Quick Pose Actions
Provides one-click editing operations for the current frame:
- **Copy / Paste Pose**: Applies copied poses across frames strictly respecting the active body mask.
- **Mirror Pose**: Horizontally flips the posture across the sagittal plane (ideal for reverse-step animation).
- **Smooth Frame**: Slerps current pose with neighboring frames to dampen isolated twitches.
- **Reset Frame**: Restores the raw estimated pose.
- **T-Pose**: Resets to canonical reference T-pose.

### 4.3 8-Slot Pose Palette & Blending
Provides 8 fast-access pose storage slots.
- Record key poses using `💾 Store to Slot`.
- Blend stored poses into target frames with customizable `Blend Weight` (0.0 to 1.0) and body part masks.

---

## 5. [Tab 2: Repair] Geometric Constraints, Grounding & Glitch Fix

### 5.1 Additive Range Offset
Adds continuous offsets across the active In-Out range:
- `Hips Y Offset`: Adjusts character height in meters.
- `Arm Open Angle`: Symmetrically widens or narrows shoulder abduction.
- `Fade at Range Edges`: Applies smooth Hermite falloff at range boundaries to eliminate abrupt step changes.

### 5.2 Armpit & Chest Penetration Limiter
Guarantees a minimum arm opening angle (`Min Armpit Angle`) for upper arms, preventing hands and forearms from clipping through the chest or torso geometry.

### 5.3 Foot Grounding & Sliding Prevention (Ground Snap & Lock Feet)
- **Foot Grounding**: Evaluates forward kinematics (FK) to measure foot and ankle world heights. If either foot penetrates below `Ground Plane Y`, pelvis height is adjusted upward to ensure firm contact.
- **Lock Feet**: Fixes the ankle's 3D coordinates to the In-frame contact position throughout the range using 2-Bone IK, eliminating foot-sliding artifacts.

### 5.4 Glitch Highlighter & Auto-Fixer
Scans motion data for unnatural tracking artifacts based on high angular velocity (>720°/s) and sharp isolated reversals (>110° spikes).
- Displays warning markers directly on the timeline track.
- The `Go` button navigates directly to the corrupted frame.
- The `⚡ Fix All Glitches` button repairs all detected anomalies in a single atomic operation via neighboring-frame Slerp.

---

## 6. [Tab 3: Timing] Temporal & Keyframe Editing

### 6.1 In / Out Range Selection
Specifies a subset of frames for batch operations (tweens, retiming, grounding, offsets). The selected range is highlighted with an aqua-tinted band on the timeline ruler.

### 6.2 Range Tweening & Interpolation
Interpolates a missing or corrupted span of frames between the `In` and `Out` anchor frames using selected easing curves.
- **Supported Easing**: `Linear`, `EaseIn` ($t^2$), `EaseOut` ($1 - (1-t)^2$), `EaseInOut`, and `SmoothStep` ($3t^2 - 2t^3$).
- Rotations are blended via spherical linear interpolation (`Quaternion.Slerp`), and root displacement is interpolated via vector `Lerp`.

### 6.3 Range Retiming (Time Warp)
Stretches or compresses the duration of the selected In-Out frame range to a target frame count while resampling intermediate poses with continuous Slerp blending. Fast presets (1.5x, 2x, 0.5x) allow instant pacing adjustments.

### 6.4 Seamless Loop Boundary Blender
Eliminates seam discontinuities in looping animations (e.g. idle or run cycles) by cross-fading trailing frames into the opening frame (Frame 0). Handles root position continuity in InPlace playback mode.

---

## 7. [Tab 4: Polish] Stylized Hand-Keyed Polish

Monocular AI pose estimation models typically generate dense, uniform-velocity keyframes that lack the dynamic spacing, timing contrast, and physical weight of handcrafted character animation. Characters often exhibit robotic simultaneity (moving and stopping all limbs in the same frame) and floating sensations caused by weak ground contact responses.

TexMotion addresses this with an integrated suite of 9 kinematic and mathematical filters rooted in the 12 Principles of Animation and stylized Japanese limited animation techniques, elevating raw mocap into punchy, expressive performance.

![Stylized Hand-Keyed Polish Guide](images/timeline_editor_stylized_polish.jpg)

### 7.1 Presets
Three curated presets provide instant one-click configurations:

| Preset | Key Characteristics & Target Settings | Recommended Use Cases |
| :--- | :--- | :--- |
| **⚡ Action** | Snap Intensity 0.70, Pose Exaggeration 1.28x, Landing Cushion 4.5cm, Dynamic Anime Frame Stepping | High-energy game actions, acrobatics, stylized combat, dynamic dance |
| **🏋️ Weight** | Landing Cushion 4.0cm (5f recovery), Contrapposto 1.30x, Kinematic Drag 0.9f, Overshoot 0.30 | Realistic locomotion, heavy carrying, weight transfers, grounded walks |
| **🍃 Subtle** | Snap Intensity 0.25, Pose Exaggeration 1.08x, Landing Cushion 2.0cm, Arc Smoothing 0.50 | Gentle cleanup that removes camera jitter and floating while preserving organic realism |

### 7.2 Technical Specifications

#### 1. Timing & Spacing
- **Snap & Ease (Moving Holds & Punchy Transitions)** `[Default: ON]`
  - Identifies frames with angular velocity below `Hold Threshold` (default 16°/s) as moving holds with subtle breathing drift. Respaces high-velocity transition spans using cubic Hermite S-curves (SmoothStep), transforming linear mocap into crisp bursts and decisive poses.
- **Anime Frame Stepping (Limited Animation)** `[Default: OFF]`
  - Emulates Japanese anime cell animation by adaptively stepping frame poses based on velocity: 1s (30 fps) for rapid action, 2s (15 fps) for standard movement, and 3s (10 fps) for holds (Strict 2s and Strict 3s modes are also supported).
- **Keyframe Decimator (Curve Simplification)** `[Default: OFF]`
  - Employs an $SO(3)$ spherical extension of the Ramer-Douglas-Peucker algorithm to preserve only extreme keyframes and breakdowns within a specified angular tolerance (`Tolerance`, default 2.5°), replacing intermediate keys with smooth spline interpolation.

#### 2. Weight & Physics
- **Landing Cushion & Bounce** `[Default: ON]`
  - Evaluates forward kinematics (FK) vertical velocities to identify foot touch-down events. Applies a downward compression impulse (`Cushion Depth`, default 3.5cm) to pelvis `RootPositions.y` and settles back to equilibrium via a damped harmonic oscillator across `Recovery Frames` (default 4f).
- **Contrapposto Booster** `[Default: ON]`
  - Detects weight-bearing support legs from foot height and ground proximity. Tilts pelvis roll upward on the loaded side and applies a counter-tilt to `Spine1`/`Spine2` to preserve head balance, enforcing classical contrapposto posture.

#### 3. Overlapping & Drag
- **Kinematic Drag & Follow-Through** `[Default: ON]`
  - Introduces organic sub-frame phase delays (`Drag Delay`, default 0.85f) to distal joints along the anatomical kinematic chain (`Pelvis` → `Spine` → `Shoulder`/`Neck` → `Elbow`/`Head` → `Wrist`). Eliminates robotic limb simultaneity and creates whip-like fluidity.
- **Overshoot & Settling** `[Default: ON]`
  - Detects sharp deceleration events and adds an inertial overshoot past the target pose, damped out with a decaying spring response (`Overshoot Amount`, default 0.35 across `Settle Frames`, default 3f).

#### 4. Pose & Silhouette
- **Pose Exaggeration (Dynamic Push)** `[Default: ON]`
  - Extrapolates angular displacement away from the neutral reference posture by `Exaggeration Scale` (default 1.18x). Limb bends and torso twists are deepened, expanding the silhouette for maximum visual readability.
- **Trajectory Arc Smoother** `[Default: ON]`
  - Smooths 3D world-space wrist and ankle position trajectories across a Gaussian-weighted window (`Window Size`, default 5f). Analytical 2-Bone IK is re-evaluated to guide limbs along graceful spatial arcs.

### 7.3 Workflow & Best Practices
1. Define the target frame range using `[Set In]` / `[Set Out]`, or select `Entire Clip`.
2. Optionally narrow editing scope using `[Body Mask]` (e.g. Upper Body or Arms).
3. Select an initial preset (`[Action]`, `[Weight]`, or `[Subtle]`).
4. Fine-tune critical parameters such as `Exaggeration Scale` or `Cushion Depth`.
5. Click `[✨ Polish Motion]` to apply all active filters atomically.
6. Use Undo (`Ctrl+Z` / `[↩️ Undo]`) and Redo (`Ctrl+Y` / `[↪️ Redo]`) to compare before and after playback.
