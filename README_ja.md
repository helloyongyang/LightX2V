<div align="center" style="font-family: charter;">
  <h1>⚡️ LightX2V:<br> 軽量ビデオ生成推論フレームワーク</h1>

<img alt="logo" src="assets/img_lightx2v.png" width=75%></img>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ModelTC/lightx2v)
[![Doc](https://img.shields.io/badge/docs-English-99cc2)](https://lightx2v-en.readthedocs.io/en/latest)
[![Doc](https://img.shields.io/badge/ドキュメント-日本語-99cc2)](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest)
[![Docker](https://badgen.net/badge/icon/docker?icon=docker&label)](https://hub.docker.com/r/lightx2v/lightx2v/tags)

**\[ [English](README.md) | [中文](README_zh.md) | 日本語 \]**
</div>

--------------------------------------------------------------------------------

**LightX2V** は、複数の先進的なビデオ生成推論技術を組み合わせた 軽量ビデオ生成推論フレームワーク です。単一のプラットフォームで テキストからビデオ (T2V)、画像からビデオ (I2V) など多様な生成タスクとモデルをサポートします。**X2V は「さまざまな入力モダリティ（テキスト・画像など）をビデオに変換する」ことを意味します。**

## 💡 はじめに

詳細手順はドキュメントをご覧ください：**[English Docs](https://lightx2v-en.readthedocs.io/en/latest/)** | **[中文文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/)**

## 🤖 対応モデル一覧

- ✅ [HunyuanVideo-T2V](https://huggingface.co/tencent/HunyuanVideo)
- ✅ [HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)
- ✅ [Wan 2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- ✅ [Wan 2.1-I2V](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- ✅ [Wan 2.1-T2V-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) (おすすめ 🚀🚀🚀)
- ✅ [Wan 2.1-T2V-CausVid](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-CausVid)
- ✅ [SkyReels-V2-DF](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P)
- ✅ [CogVideoX 1.5-5B-T2V](https://huggingface.co/THUDM/CogVideoX1.5-5B)

## 🧾 コントリビューションガイドライン

プロジェクト全体でコードフォーマットを統一するため、`pre-commit` フックを用意しています。

> [!Tip]
> 1. 依存パッケージをインストール
>    ```bash
>    pip install ruff pre-commit
>    ```
> 2. コミット前に実行
>    ```bash
>    pre-commit run --all-files
>    ```

ご協力ありがとうございます！

## 🤝 謝辞

本リポジトリの実装は、上記すべてのモデル関連リポジトリを参考にしています。

## 🌟 Star 推移

[![Star History Chart](https://api.star-history.com/svg?repos=ModelTC/lightx2v&type=Timeline)](https://star-history.com/#ModelTC/lightx2v&Timeline)

## ✏️ 引用

本フレームワークが研究に役立った場合は、以下を引用してください。

```bibtex
@misc{lightx2v,
  author    = {lightx2v contributors},
  title     = {LightX2V: Light Video Generation Inference Framework},
  year      = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
