# 逆水寒関連の参考技術とTexMotionローカル導入候補：一次資料調査

調査日：2026-09-20。対象は公開論文、著者・開発元のページ、公式コードとライセンス。短時間の公開資料調査であり、TexMotionへの実装・動作検証は実施していない。

## 判断の要点

TexMotionの表情・頭部姿勢のローカル取得にはMediaPipe Face Landmarkerを最初の検証候補とする。全身の動画復元と接地補正はWHAM、文章から動作をつなぐ参考はTEACHであり、それぞれ入力も利用条件も異なる。これは以下の資料からの導入上の判断であり、実測結果ではない。

**本調査では、指定された8技術のいずれについても「逆水寒の特定機能・製品版に採用された」と直接示す一次資料を確認していない。** NetEase所属著者、NetEaseの公開リポジトリ、研究チームの成果一覧は研究の出自を示すが、製品採用の証拠としては扱わない。未確認は不採用を意味しない。

## 1. MediaPipe Face Landmarker：BlendShapeと頭部行列

- 確認事実：公式ガイドは478個の3D顔ランドマーク、52個のBlendShapeスコア、顔の変換行列を説明している。`output_face_blendshapes` と `output_facial_transformation_matrixes` は既定で無効。行列は標準顔モデルから検出顔への変換であり、そのまま任意のアバターの首ボーン回転になるとは説明されていない。画像・動画・ライブ入力に対応し、平滑化は `num_faces=1` の場合に適用される。[公式ガイド](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- 導入判断：表情名と対象リグの対応表、頭部の基準姿勢・座標軸・親ボーンを考慮した変換を実装し、正面／横向き／遮蔽／顔消失で評価する。52係数が対象リグに無調整で互換になるとは仮定しない。これはTexMotion向けの提案である。
- ライセンス：コードはApache-2.0。再配布時のライセンス・必要な告知の保持、変更表示等を確認する。ページ本文のCC BY 4.0をモデル重みの利用条件と混同しない。採用するモデルバンドルの配布条件は別途固定・記録する。[公式LICENSE](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- 逆水寒採用：未検証。

## 2. WHAM：contact-aware trajectory refinement

- 確認事実：対象は **World-grounded Humans with Accurate Motion**（CVPR 2024）である。同名のMicrosoftの生成モデル等ではない。単眼動画から世界座標の人体運動を復元し、接地確率が閾値を超える爪先・踵の平均速度を用いてルート速度を補正した後、学習した軌道補正器で姿勢・速度を更新する。平面地面への限定を避け、階段等も扱う設計である。[論文・接地補正節](https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.pdf)
- 導入判断：まず録画動画の全身復元を評価する候補。接地確率を利用した足滑り補正の設計も参考になるが、既存の任意スケルトンに補正器だけを直接差し込める保証はない。公開デモは動画処理であり、TexMotionでのライブ遅延は未測定。
- 実装・条件：公式コードにはSMPL取得の登録手順、SLAMを省略する `--estimate_local_only`、追加のTemporal SMPLify処理が記載される。[公式リポジトリ](https://github.com/yohanshin/WHAM)
- ライセンス：確認時点のコードLICENSEはMIT。論文概要の「研究目的で公開」という記載だけからコードを非商用限定と断定しない。一方、SMPL、モデル重み、データセット、SLAM等の依存物にMITが一括適用されるわけではなく、それぞれの条件確認が残る。[コードLICENSE](https://github.com/yohanshin/WHAM/blob/main/LICENSE)／[論文概要](https://arxiv.org/abs/2312.07531)
- 逆水寒採用：未検証。

## 3. TEACH：文章に従う時系列の動作合成

- 確認事実：TEACH（3DV 2022）は複数の自然言語指示の順番に従って3D人体動作を生成する。BABELの動作・文章データを利用し、動作内は非自己回帰、動作間は自己回帰のTransformer構成を採る。[著者プロジェクト](https://teach.is.tue.mpg.de/)／[論文](https://arxiv.org/abs/2209.04066)
- ローカル候補：公式実装は文章と継続時間を受けるデモ、身体頂点のnpyと動画出力を説明する。Ubuntu 20.04、Python 3.9以上での試験が記載され、SMPLH等が必要。Windowsでの実行性やTexMotionのボーンへの変換は別途検証する。[公式コード](https://github.com/atnikos/teach)
- ライセンス：公式READMEは**非商用の科学研究目的**と明記し、商用ライセンスの問い合わせ先を示す。依存ソフト・データにも個別条件がある。ローカル実行できることを製品組み込み許可とみなさない。[READMEのLicense節](https://github.com/atnikos/teach#license)
- 逆水寒採用：未検証。

## 4. MoCap-Solver：光学マーカーのノイズ除去

- 確認事実：SIGGRAPH 2021の研究。NetEase-GameAIの公式コードは、生の光学モーションキャプチャ・マーカーから、きれいなマーカーとスケルトン動作を求めると説明する。RGB動画から直接姿勢を推定する方式ではない。[公式コード・論文書誌](https://github.com/NetEase-GameAI/MoCap-Solver)
- 導入判断：光学マーカー入力を扱う場合の候補。READMEの例は56マーカー・24関節で、学習には対応する正解とバインド姿勢が必要。旧Python/CUDA等の環境指定があるため、即時導入の優先度は低い。これらは公開実装の条件であり、任意入力の対応保証ではない。
- ライセンス：確認したルート一覧・READMEでは利用許諾の明示を確認できなかった。公開コードという理由だけで商用利用・再配布可と判断しない。SMPL、AMASS、SURREAL等の依存資産も個別に確認する。[公式README](https://github.com/NetEase-GameAI/MoCap-Solver)
- 逆水寒採用：NetEase組織による公開は確認できるが、逆水寒での採用は未検証。

## 5. 3DStyFace：名称の同定が未完了

`3DStyFace`、`3D StyFace`、NetEase／网易／伏羲との組み合わせを検索したが、今回の範囲では当該名称の論文・公式実装・製品説明を同定できなかった。従って、技術内容、一次資料URL、ライセンス、逆水寒採用はいずれも**未確認**とする。一次資料URLを作り上げたり、3DStyleNetなど別名の研究を同一視したりしない。

参考として、FuxiCVの研究一覧には顔再構成、Face-to-Parameter Translation、FDN等が掲載されるが、これは3DStyFaceの同定・採用証拠にはならない。[研究チームの公開一覧](https://fuxicv.github.io/fuxicv/)

次の調査には、元の表記が載った記事・講演資料・スクリーンショットの出所が必要。正式名称を確認するまでTexMotionの導入候補に数えない。

## 6. FDN：頭部姿勢推定

- 確認事実：**Feature Decoupling Network for Head Pose Estimation**（AAAI 2020）。単一RGB画像からランドマークを使わず頭部姿勢を推定する3分岐ネットワークで、著者にNetEase Fuxi AI Lab所属のYi Yuanが含まれる。[AAAI公式論文ページ](https://ojs.aaai.org/index.php/AAAI/article/view/6974)
- 導入判断：頭部姿勢推定の比較研究。表情BlendShapeの出力とは別の課題であり、Face Landmarkerの機能全体の代替とは扱わない。
- ライセンス：今回、当該研究の公式コード・重みの明示的利用許諾を確認できなかった。論文が読めることと実装・重みの利用許諾は別。コード導入は保留。
- 逆水寒採用：著者所属は確認済み、製品採用は未検証。

## 7. I²R-Net / I2R-Net：複数人物の2D姿勢推定

- 確認事実：**Intra- and Inter-Human Relation Network for Multi-Person Pose Estimation**（IJCAI 2022）。人物内と人物間の関係を利用し、キーポイントのヒートマップを予測する。COCO、CrowdPose、OCHumanを評価に用いる。人体3D復元・接地補正を直接行う研究としては扱わない。[IJCAI公式論文](https://www.ijcai.org/proceedings/2022/0120.pdf)
- 導入判断：複数人物や遮蔽のある入力の2D検出候補。そこからTexMotion用の3D動作を得る処理は別途必要。[公開実装](https://github.com/leijue222/Intra-and-Inter-Human-Relation-Network-for-MPEE)
- ライセンス：公開実装のLICENSEはMITで、著作権・許諾表示の保持が必要。重み・データ・依存物の条件も確認する。[LICENSE](https://github.com/leijue222/Intra-and-Inter-Human-Relation-Network-for-MPEE/blob/main/LICENSE)
- 逆水寒採用：未検証。

## 8. EA-RAS：解剖学的骨格の復元

- 確認事実：**Towards Efficient and Accurate End-to-End Reconstruction of Anatomical Skeleton**（2024）。単一RGB画像から解剖学的骨格を推定し、人体表面モデルも推定する方式。著者サイトはNetease Fuxi Robotと明記する。[論文](https://arxiv.org/abs/2409.01555)／[著者プロジェクト](https://ea-ras.github.io/)
- 導入判断：研究の目的は解剖学的骨格であり、ゲーム向けの標準ボーンアニメーションや時系列の接地保証とは異なる。論文の高速化倍率をTexMotion上のfpsに読み替えない。
- 公開・ライセンス：著者ページは確認時点でコード・Quick Startを公開予定と表示していた。コードへのリンクはあるが、利用可能な完成実装・重み・許諾を確認できていないため、導入は保留。[リンク先リポジトリ](https://github.com/ea-ras/EA-RAS)
- 逆水寒採用：研究組織との関係は確認済み、製品採用は未検証。

## TexMotionで次に検証する順序

| 優先度 | 候補 | 最小限の検証 | 残る条件 |
| --- | --- | --- | --- |
| 1 | MediaPipe Face Landmarker | ローカル動画で表情係数と頭部変換を保存し、対象リグへ適用 | モデル条件、座標変換、欠落処理、遅延 |
| 2 | WHAM | 歩行・停止・階段で世界軌道と足滑りを比較 | SMPL等の条件、環境、リターゲット |
| 3 | TEACH | 複数文章からの動作接続を研究用に評価 | 非商用研究制限、人体モデル、商用許諾 |
| 条件付き | I²R-Net / MoCap-Solver | 複数人物2D入力／光学マーカー入力が必要な場合 | 入力適合性と資産ごとの利用条件 |
| 保留 | FDN / EA-RAS / 3DStyFace | 実装公開・名称・許諾の追加確認 | 現時点で直接導入を推奨する証拠不足 |

採用時にはコードのコミット、重みのバージョンと配布元、LICENSE、依存資産を固定して記録する。本書の「未確認」は今回の調査範囲の限界を示し、非公開の製品実装や契約の有無を推測するものではない。
