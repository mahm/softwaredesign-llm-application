---
name: pptx-generator
description: generate_pptxツールを使用してPowerPointプレゼンテーションファイル(.pptx)を生成するスキル。スライド作成、プレゼンテーション生成、PowerPoint出力が必要な場合に使用する。
---

# PPTX生成スキル

## 概要

このスキルは `generate_pptx` ツールを呼び出してPowerPointファイルを生成します。
スクリプトの作成や実行は不要です。

## 使用手順

### ステップ1: スライドデータの準備

スライドのアウトラインが確定したら、以下のJSON形式でデータを整理します。

```json
{
  "title": "プレゼンテーションタイトル",
  "author": "著者名",
  "slides": [
    {
      "type": "title",
      "title": "メインタイトル",
      "subtitle": "サブタイトル"
    },
    {
      "type": "content",
      "title": "セクションタイトル",
      "bullets": ["項目1", "項目2", "項目3"]
    },
    {
      "type": "section",
      "title": "セクション区切り"
    }
  ]
}
```

### ステップ2: ツールの呼び出し

`generate_pptx` ツールに上記のデータを渡して呼び出します。デザインやレイアウトはフロントエンド側で自動的に適用されるため、データだけ渡せば完了です。
