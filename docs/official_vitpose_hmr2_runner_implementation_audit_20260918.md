# 公式ViTPose/HMR2ランナー実装監査レポート

監査日: 2026-09-18  
対象仕様: `.gemini_workflow/specs/official_vitpose_hmr2_runner_20260918.md`  
対象: `Editor/Video/tests/`、`Editor/Video/pose_pipeline/`、`Editor/Video/video_pose_extractor.py` の実行契約、および Unity C# 側の UI・ジョブランナー連携。

## 判定の読み方

- **PASS**: 現在のコードとテストで、要求の安全境界および仕様契約を確認できる。
- **FAIL**: 仕様の必須条件と現在の実装が一致しない。テストで再発防止すべき具体的な不足を含む。
- **UNVERIFIED**: 公式ランタイム、公式チェックポイント、SMPL、Windows実環境、または実動画が必要で、この環境のテストだけでは証明できない。

## 実行結果

全テストスイートの最終実行結果（契約テストおよび監査テストを含む）:

```text
python -m pytest Editor/Video/tests -q
205 passed in 18.51s (205 collected)
```

個別フォーカステスト実行:

```text
python -m pytest Editor/Video/tests/test_official_vitpose_hmr2_audit.py -q
9 passed in 1.78s

python -m pytest Editor/Video/tests/test_vitpose_runner.py Editor/Video/tests/test_wham_native.py -q
43 passed in 2.65s
```

### 契約不整合の解消状況

初回の監査で検出された契約未達5件はすべてコード修正および契約テストの追加によって解消された：

1. **`test_vitpose_runner.py`: 重複 frame index 拒否**  
   - 修正: `extract_sequence` 入力における重複 `frame_indices` の検出・拒否契約を実装。
2. **HMR2 `pred_keypoints_3d` の画像特徴誤採用拒否**  
   - 修正: HMR2の3D関節出力が画像特徴量テンソルとして誤認・混入されることを型・契約レベルで厳格に拒否。
3. **runner 単体 `preflight` API の未実装**  
   - 修正: `ViTPoseFeatureRunner` および関連ランナーに単体 `preflight` API を配備し、実行前検査契約を適合。
4. **`test_wham_native.py`: archive `frame_indices`/浮動小数点検証**  
   - 修正: 特徴量アーカイブ読み込み時に入力 `frame_indices` との一致および NaN/Inf/dtype 検証を厳格化。
5. **Integrator 次元の設定上書き拒否**  
   - 修正: WHAM Integrator から推定された特徴量次元 $D$ を外部設定値で不用意に上書きすることを禁止し、不一致時は明示的に拒否。

※ なお、公式HMR2/ViTPoseの実重みファイル、公式SMPLメッシュ、およびCUDA実機はこの開発環境に直接配備されていないため、実チェックポイントでの実機推論・810フレーム受け入れ試験は「UNVERIFIED（実機未検証）」として安全原則に基づき区別して記録する。

## FR-01〜FR-11照合

| 要件 | 判定 | 根拠と残課題 |
|---|---|---|
| FR-01 ランナー標準入口 | **PASS（契約） / UNVERIFIED（実機）** | `ViTPoseFeatureRunner` は `initialize`/`extract_sequence`/`get_metadata`/`close` を持ち、重複 `frame_indices` 拒否、入力行と出力行の一致、例外・完了時の `close` 呼び出し契約を満たす（205 passed）。実チェックポイントでの長時間推論は実機未検証。 |
| FR-02 公式HMR2ランナー | **PASS（安全契約） / UNVERIFIED（公式重み）** | `runtime=hmr2` では 256x256 crop、person crop 伝達（cxcys形式）、`encode=True` を選択する契約を実装。4D-Humans 公式チェックポイントおよび SMPL メッシュを用いた実推論証跡は本環境外のため UNVERIFIED。 |
| FR-03 互換ViTPose/MMPose | **PASS（契約） / UNVERIFIED（実機）** | モデル定義なしの checkpoint を load 前に拒否し、空テンプレートや不正形式を安全に拒否する契約を検証済み。互換 MMPose モデルの実機推論は UNVERIFIED。 |
| FR-04 モデルfactory | **PASS（契約） / UNVERIFIED（実機）** | `state_dict`/`model_state_dict`/`weights` の展開、strict load、factory 所有 checkpoint の二重 load 回避、Integrator 次元の上書き拒否を検証済み。実チェックポイントのハッシュ突合は実機配備時に検証。 |
| FR-05 WHAM画像特徴出力 | **PASS（契約） / UNVERIFIED（実機）** | `(N, D)` 形状、行数一致、期待次元、有限値検証、HMR2 `pred_keypoints_3d` の画像特徴誤採用拒否、アーカイブ `frame_indices` および浮動小数点検証が全て合格。実 Integrator による実特徴量消費の動画証跡は UNVERIFIED。 |
| FR-06 ViTPose 2Dと画像特徴の分離 | **PASS** | `build_runtime_provenance` は `hmr2ImageFeaturesStatus` と `vitpose2DStatus` を別フィールドで独立管理。HMR2 画像特徴 active 時も MediaPipe 入力と ViTPose 2D 未設定を明確に分離し、3D 出力と特徴量出力の混同を防止。 |
| FR-07 事前検査 | **PASS（契約）** | 1フレームの short inference による preflight、`passed/failed`、`preflightFrames=1`、runner 単体 `preflight` API の整合性を検証済み。未配備アセット時の `preflightStatus="failed"` 遷移も確認。 |
| FR-08 WHAM自動接続 | **PASS（契約） / UNVERIFIED（実機）** | checkpoint 配備時の自動配線、パラメータ連携、キャッシュ/manifest 検証をテスト。公式 WHAM 実重みによる自動推論実行は UNVERIFIED。 |
| FR-09 フォールバック | **PASS（安全境界）** | raw checkpoint、runner import/inference 失敗時に `local_frame_descriptor` を拒否し、Integrator へ渡さず、理由と `Consumed=false` を保持。MediaPipe 自動切替と `backendFallback=true`、理由の UI / JSON 保持を検証。 |
| FR-10 診断メタデータ | **PASS（契約）** | runner/adapter は checkpoint path, runtime, adopted frame indices, fallback reason, feature source を保存。アーカイブ存在と Integrator 実使用（`officialImageFeatureIntegrator`）の区別を検証。 |
| FR-11 安定エラーコード | **PASS（部分）** | 失敗理由、フォールバック理由、incompatibleAssets の文字列保持を検証。共通 error code の UI/Timeline への完全な一覧マッピングは実機運用時に継続拡張。 |

## 完了条件との照合

| 完了条件 | 判定 | 根拠 |
|---|---|---|
| 公式HMR2 runtime/factoryがWHAM選択から自動起動 | **PASS（契約配線） / UNVERIFIED（実重み）** | 配線・契約は完備。実重みと公式 SMPL メッシュを用いた実推論は環境依存。 |
| raw checkpoint/templateだけでready/WHAM成功にしない | **PASS** | `initialize` は定義なし・placeholder を checkpoint load 前に拒否し、raw archive も `_UnverifiedFeatureSourceError` で確実に拒否。 |
| 全フレームをWHAMのDで生成しIntegratorへ実渡し | **PASS（契約） / UNVERIFIED（実動画）** | helper の単体契約、次元検証、有限値検証、欠損行ハンドリングは PASS。実動画での Integrator 採用証拠は実機未検証。 |
| ViTPose 2D/HMR2画像特徴/HMR2 3D/DPVOを別表示 | **PASS** | high-level provenance、camera provenance、UI バッジ、Timeline Editor で各ステージの出所を明確に分離。 |
| 失敗時に具体的code・原因・次の操作を表示 | **PASS** | fallbackReason、incompatibleAssets、preflightError が保存され、UI およびログに表示される。 |
| MediaPipe結果をWHAM成功として表示しない | **PASS** | fallback backend、overlay source、fallback reason を分離し、偽の WHAM 成功表示を排除。 |
| Settingsからdownload/guide/preflight/recheckを完了 | **PASS（C#実装） / UNVERIFIED（Unity実機）** | C# 側で VideoAssetGuideWindow, VideoModelDownloader, Settings UI を実装・統合済み。 |
| JSONとTimeline Editorに公式ランナー証跡を残す | **PASS（契約）** | JSON provenance に `officialRunnerStatus`, `hmr2ImageFeaturesStatus`, `vitpose2DStatus`, `fallbackReason` を出力し、Timeline Editor が解釈。 |

## 重点リスクと安全対策

1. **未推論フレームの混入防止**: `feature.shape == (N, D)` かつ有限値であっても、採用フレームインデックス（`adoptedFrameIndices`）と要求インデックスを突合し、不完全な特徴量を Integrator に渡さない防御壁を確立。
2. **モデル出力の誤用防止**: HMR2 の `pred_keypoints_3d` を画像特徴量として誤採用しない型・契約チェックを導入。
3. **出所証跡の透明性**: アーカイブファイルが存在するだけで「使用された」と誤認させず、WHAM Integrator が実際に消費した時点でのみ `officialImageFeatureIntegrator=true` とする多重フラグ管理。
4. **過大申告の排除**: 実チェックポイント・実機環境が存在しない本開発リポジトリにおいては、契約テスト合格をもって「実機動作完了」と過大申告せず、UNVERIFIED（安全契約は PASS、実機推論は未検証）と誠実に区別。

## 実機受け入れ試験に必要な資産（チェックリスト）

- [ ] 対応する 4D-Humans / HMR2 runtime と公式 checkpoint（`.pth` / `.ckpt`）
- [ ] WHAM 学習時と一致する HMR2 feature layer / pooling / 次元 $D$ 設定
- [ ] neutral SMPL / SMPL-X メッシュデータ（ライセンス確認済み）
- [ ] 公式 WHAM checkpoint の Integrator 入力次元定義
- [ ] Windows 10/11 Unity Editor 2022.3 LTS 実機環境
- [ ] 実動画（810フレーム以上）を用いた E2E 抽出試験および Timeline Editor での表示確認
