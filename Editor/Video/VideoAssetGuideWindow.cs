using System.IO;
using UnityEditor;
using UnityEngine;

namespace TexMotion.Editor
{
    /// <summary>
    /// Small, metadata-driven setup guide for manual or partially manual
    /// Video2Motion assets. It is intentionally an EditorWindow rather than a
    /// browser-only help link so the exact local destination and Browse setting
    /// remain visible while the user provisions an upstream asset.
    /// </summary>
    public sealed class VideoAssetGuideWindow : EditorWindow
    {
        private VideoModelDefinition _asset;
        private TexMotionSettings _settings;
        private Vector2 _scroll;

        private bool IsJapanese => (_settings ?? TexMotionSettings.instance) != null &&
            (_settings ?? TexMotionSettings.instance).Language == TexMotionLanguage.Japanese;

        public static void Show(VideoModelDefinition asset, TexMotionSettings settings = null)
        {
            if (asset == null) return;
            var window = GetWindow<VideoAssetGuideWindow>(false, "Video Asset Guide");
            window._asset = asset;
            window._settings = settings ?? TexMotionSettings.instance;
            window.minSize = new Vector2(520f, 460f);
            window.Show();
            window.Focus();
        }

        /// <summary>
        /// Expands catalog path tokens for display and keeps this transformation
        /// pure so static contract tests can verify it without opening Unity UI.
        /// </summary>
        public static string ResolveRecommendedInstallDirectory(
            VideoAssetGuide guide,
            TexMotionSettings settings)
        {
            if (guide == null || string.IsNullOrWhiteSpace(guide.RecommendedInstallDirectory))
                return string.Empty;

            string value = guide.RecommendedInstallDirectory.Trim();
            string modelDirectory = settings == null
                ? "<Video2Motion model directory>"
                : settings.GetEffectiveVideoModelDirectory();
            string projectRoot = settings == null
                ? "<Unity project root>"
                : settings.GetProjectRootDirectory();
            value = value.Replace("{VideoModelDirectory}", modelDirectory)
                .Replace("{ProjectRoot}", projectRoot);
            return value.Replace('/', Path.DirectorySeparatorChar);
        }

        private void OnGUI()
        {
            _settings = _settings ?? TexMotionSettings.instance;
            titleContent = new GUIContent(TexMotionLocalization.TrLiteral("Video Asset Guide"));
            VideoAssetGuide guide = VideoModelCatalog.GetAssetGuide(_asset);
            if (_asset == null || guide == null)
            {
                EditorGUILayout.HelpBox(
                    TexMotionLocalization.TrLiteral("This catalog row has no manual setup guide."),
                    MessageType.Info);
                if (GUILayout.Button(TexMotionLocalization.TrLiteral("Close"))) Close();
                return;
            }

            DrawLanguageToolbar();
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            EditorGUILayout.LabelField(_asset.DisplayName ?? _asset.Id, EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("Catalog id: ") + (_asset.Id ?? "<unknown>") +
                TexMotionLocalization.TrLiteral("  •  Role: ") + _asset.AssetRole,
                EditorStyles.miniLabel);

            EditorGUILayout.Space(6f);
            EditorGUILayout.LabelField(
                TexMotionLocalization.TrLiteral("License / manual reason"),
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                GetLicenseReason(guide),
                _asset.IsManualProvisioning ? MessageType.Warning : MessageType.Info);

            DrawValue(TexMotionLocalization.TrLiteral("Official download / registration"), guide.OfficialUrl);
            if (!string.IsNullOrWhiteSpace(guide.OfficialUrl) &&
                GUILayout.Button(
                    TexMotionLocalization.TrLiteral("Open official download / registration page"),
                    EditorStyles.miniButton))
            {
                Application.OpenURL(guide.OfficialUrl);
            }

            DrawValue(
                TexMotionLocalization.TrLiteral("Recommended install directory"),
                ResolveRecommendedInstallDirectory(guide, _settings));
            DrawValue(TexMotionLocalization.TrLiteral("Required file or directory"), guide.RequiredFileName);
            DrawValue(TexMotionLocalization.TrLiteral("Browse setting"), guide.BrowseSetting);

            EditorGUILayout.Space(6f);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("Preflight steps"), EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                GetPreflightSteps(guide),
                MessageType.None);

            EditorGUILayout.Space(6f);
            EditorGUILayout.LabelField(TexMotionLocalization.TrLiteral("What to do next"), EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                IsJapanese
                    ? "上記の参照設定でローカルのファイル／フォルダーを指定してください。次にランタイム事前チェックを実行します。Ready（準備完了）の行は、そのロールに必要な形式とエントリーポイントの検証に成功したことを示します。"
                    : "Use the Browse setting above to point TexMotion at the local file/folder. "+
                      "Then run Runtime Preflight; a green Ready row means the role-specific "+
                      "format and entry-point checks passed.",
                EditorStyles.wordWrappedLabel);

            EditorGUILayout.Space(10f);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Open TexMotion Settings"), GUILayout.Height(26f)))
            {
                TexMotionWindow.ShowSettingsWindow();
            }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Copy install directory"), GUILayout.Height(26f)))
            {
                EditorGUIUtility.systemCopyBuffer = ResolveRecommendedInstallDirectory(guide, _settings);
            }
            if (GUILayout.Button(TexMotionLocalization.TrLiteral("Close"), GUILayout.Height(26f))) Close();
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.EndScrollView();
        }

        private void DrawLanguageToolbar()
        {
            var settings = _settings ?? TexMotionSettings.instance;
            if (settings == null) return;

            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.Label(TexMotionLocalization.Tr(TexMotionLocalization.Language), EditorStyles.miniLabel);
            GUILayout.FlexibleSpace();
            string[] options =
            {
                TexMotionLocalization.Tr(TexMotionLocalization.English),
                TexMotionLocalization.Tr(TexMotionLocalization.Japanese)
            };
            int selected = settings.Language == TexMotionLanguage.Japanese ? 1 : 0;
            int next = EditorGUILayout.Popup(selected, options, EditorStyles.toolbarPopup, GUILayout.Width(110f));
            if (next != selected)
            {
                settings.Language = next == 1 ? TexMotionLanguage.Japanese : TexMotionLanguage.English;
                settings.Save();
                Repaint();
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(4f);
        }

        private string GetPreflightSteps(VideoAssetGuide guide)
        {
            if (guide == null) return string.Empty;
            if (!IsJapanese)
            {
                return string.IsNullOrWhiteSpace(guide.PreflightSteps)
                    ? "Open Settings > Video2Motion and run Runtime Preflight after provisioning."
                    : guide.PreflightSteps;
            }

            switch (guide.BrowseSetting)
            {
                case "HMR2ModelPath":
                case "HMR2CheckpointPath":
                    return "HMR2アーカイブを選択し、公式ランタイムと中立SMPLボディモデルを設定してからランタイム事前チェックを実行してください。TexMotionブリッジは自動で使用されます。";
                case "HybrIKModelPath":
                    return "チェックポイントをダウンロード／選択し、互換性のあるHybrIKアダプターを設定してから抽出前にランタイム事前チェックを実行してください。";
                case "WHAMBodyModelPath":
                    return "ボディモデルを登録／ダウンロードして参照で選択し、ランタイム事前チェックを実行してください。不互換またはテンプレート警告が出た場合は置き換えます。";
                case "WHAMImageFeatureBackbonePath":
                case "ViTPoseModelPath":
                    return "互換性のあるViTPoseモデルファクトリー／ランナー、または事前計算済み特徴量アーカイブを指定してからランタイム事前チェックを実行してください。";
                case "WHAMImageFeaturePath":
                    return "設定 > Video2Motionの「ViTPose特徴量アーカイブを作成」で動画を選択し、ワンクリック抽出を実行してください。互換ランナーがない場合は、モデル定義／ファクトリーを先に設定してから再実行します。完了後にランタイム事前チェックを実行してください。";
                case "WHAMImageFeatureModelDefinitionPath":
                case "ViTPoseModelDefinitionPath":
                    return "同梱の契約テンプレートを互換性のあるcreate_modelファクトリーへ置き換え、参照で選択してからランタイム事前チェックを実行してください。";
                case "WHAMImageFeatureConfigPath":
                case "ViTPoseConfigPath":
                    return "ランナー／特徴量契約を確認し、必要に応じてテンプレートを置き換え、設定ファイルを参照で選択してからランタイム事前チェックを実行してください。";
                case "WHAMCameraModelPath":
                    return "入力動画に合わせてテンプレートのカメラ内部パラメーターを置き換え、ファイルを参照で選択してからランタイム事前チェックを実行してください。";
                case "WHAMDpvoModelPath":
                    return "チェックポイントを使用する場合は互換性のあるDPVOランタイムを導入し、チェックポイントまたは出力を参照で選択してからランタイム事前チェックを実行してください。";
                case "HMR2AdapterPath":
                    return "TexMotion同梱のHMR2ブリッジが自動で使用されます。独自ブリッジへ置き換える場合だけ.pyファイルを参照で選択してください。";
                case "HMR2RuntimePath":
                    return "指定バージョンの上流ランタイムをパッケージ外へクローン／インストールし、フォルダーを参照で選択してPython環境を設定してからランタイム事前チェックを実行してください。";
                case "ViTPoseRuntimePath":
                    return "互換性のあるViTPose／MMPoseランタイムをパッケージ外へ導入し、フォルダーを参照で選択してからランタイム事前チェックを実行してください。通常の公式HMR2経路では不要です。";
                case "HMR2BodyModelPath":
                    return "中立SMPLボディモデルを登録／ダウンロードし、.pklファイルを参照で選択してからHMR2ランタイム事前チェックを実行してください。";
                case "HybrIKAdapterPath":
                    return "create_backendとinfer_sequenceを公開するブリッジを設定し、.pyファイルを参照で選択してからランタイム事前チェックを実行してください。";
                default:
                    return "設定 > Video2Motionでローカルファイルを指定し、モデルを準備した後にランタイム事前チェックを実行してください。";
            }
        }

        private string GetLicenseReason(VideoAssetGuide guide)
        {
            if (guide == null || string.IsNullOrWhiteSpace(guide.LicenseReason))
            {
                return IsJapanese
                    ? "このステージにはユーザーによる明示的なローカル設定が必要です。"
                    : "This stage requires an explicit local setup step.";
            }
            if (!IsJapanese) return guide.LicenseReason;

            switch (guide.BrowseSetting)
            {
                case "HMR2ModelPath":
                case "HMR2CheckpointPath":
                    return "これは4D-Humansの上流アーティファクトです。TexMotionは重みを再配布・権利主張しません。公式経路には別途ライセンスされたSMPLボディとランタイムが必要です。";
                case "HybrIKModelPath":
                    return "これは上流アーティファクトであり、実行可能なブリッジは含まれません。使用前にソース、チェックポイント、ボディモデルの利用条件を個別に確認してください。";
                case "WHAMBodyModelPath":
                    return "SMPL／SMPL-Xボディデータは別途ライセンスされています。ユーザーが用意する必要があり、TexMotionは自動ダウンロードや同梱を行いません。";
                case "WHAMImageFeatureBackbonePath":
                case "ViTPoseModelPath":
                    return "チェックポイントは公開取得できますが、アーキテクチャ、ランタイム、統合条件は上流プロジェクトの責任です。生のstate-dictだけでは実行できません。";
                case "WHAMImageFeaturePath":
                    return "互換ViTPoseランナーが設定済みなら、Settingsのワンクリック処理が動画のFPS／トリムと行番号を自動整列します。上流モデル定義や重みの利用条件はユーザーが確認してください。";
                case "WHAMImageFeatureModelDefinitionPath":
                case "ViTPoseModelDefinitionPath":
                    return "パッケージのテンプレートはフックの契約だけを示します。上流ViTPoseのアーキテクチャと重みを実行可能な形では再配布していません。";
                case "WHAMImageFeatureConfigPath":
                case "ViTPoseConfigPath":
                    return "ランナー設定はメタデータであり、ViTPose本体ではありません。互換性のある上流ランナー／ファクトリーは手動で用意してください。";
                case "WHAMCameraModelPath":
                    return "カメラキャリブレーションは動画ごとに異なります。パッケージはテンプレートを置けますが、校正値やワールドモーションを保証しません。";
                case "WHAMDpvoModelPath":
                    return "チェックポイントだけではカメラモーションランナーになりません。DPVOが利用できない場合は、TexMotionが使うローカルフォールバックを結果に明示します。";
                case "HMR2AdapterPath":
                    return "HMR2ブリッジはTexMotionに同梱されています。公式ランタイム、チェックポイント、別途ライセンスされたSMPLボディはユーザーが用意してください。";
                case "HMR2RuntimePath":
                    return "公式HMR2ソース、ランタイム、依存関係は第三者研究ソフトウェアです。確認できない条件のままTexMotionが同梱・自動インストールすることはありません。";
                case "ViTPoseRuntimePath":
                    return "互換ViTPose／MMPoseランタイムとそのモデル定義は上流ソフトウェアです。公式HMR2の既定経路では要求されません。";
                case "HMR2BodyModelPath":
                    return "SMPLは別途ライセンスされています。公式HMR2ランタイムにはユーザーが取得・登録したボディモデルが必要で、TexMotionは自動取得しません。";
                case "HybrIKAdapterPath":
                    return "HybrIKのソース、ランタイム、チェックポイントの利用条件は上流プロジェクトの責任です。生のチェックポイントを実行可能とみなさず、明示的なローカルアダプターを使用します。";
                default:
                    return "このアセットは上流条件により手動／ローカルで準備する必要があります。";
            }
        }

        private static void DrawValue(string label, string value)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(205f));
            EditorGUILayout.SelectableLabel(
                string.IsNullOrWhiteSpace(value)
                    ? TexMotionLocalization.TrLiteral("<not specified>")
                    : value,
                EditorStyles.textField,
                GUILayout.Height(18f));
            EditorGUILayout.EndHorizontal();
        }
    }
}
