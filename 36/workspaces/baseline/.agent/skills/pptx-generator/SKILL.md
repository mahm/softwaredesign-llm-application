---
name: pptx-generator
description: arXiv論文URLからスライドを作成する。論文を取得・分析し、スライド構成JSONを保存、ユーザー確認後にgenerate_pptxツールでスライドを生成する。
---

# arXiv論文スライド生成スキル

## ワークフロー

1. arXiv IDをURLから抽出する(例: 2301.00001)
2. `curl -sL https://arxiv.org/html/{id}` でHTML本文を取得する。本文が取得できない場合や数百字程度のスタブしか返らない場合(2023年以前の論文に多い)は、`curl -sL https://ar5iv.labs.arxiv.org/html/{id}` で全文を取得する
3. 論文を分析し、スライド構成を次のJSON形式で `./slides/{arXiv ID}.json`(例: `./slides/2603.03303.json`)に保存する
4. アウトラインをユーザーに提示し、確認を得る
5. 確認後、`generate_pptx`ツールにファイルパス(例: `./slides/2603.03303.json`)を渡してスライドを生成する。バリデーションエラーが返された場合は、エラー内容に基づいてJSONファイルを修正し、再度ツールを呼び出す

## スライド構成JSONの形式

スライド枚数はタイトルスライド1枚 + コンテンツスライド5枚 = 計6枚とする。

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
