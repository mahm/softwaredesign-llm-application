---
name: acp-multi-agent
description: ACPXを使ってClaude CodeとCodexへ読み取り専用の調査を依頼し、人間の回答を挟んで最終レポートをまとめる
---

# ACPマルチエージェント調査

## 使う場面

対象リポジトリについて、Claude CodeとCodexの両方に独立した見立てを出してもらうときに使う。

## 手順

1. `ask_acp_agents` を `turn=1` で呼び出し、Claude CodeとCodexを同時に起動する。
2. セッション名はClaude Codeを `claude-`、Codexを `codex-` で始める。
3. 初回の結果から、対象リポジトリの調査に関係する確認質問だけを取り出す。
4. 確認質問がある場合は、`ask_human` を1回だけ呼び出してまとめて確認する。
5. ユーザーの回答だけを `humanAnswer` に入れ、同じセッション名で `ask_acp_agents` を次の `turn` として呼び出す。
6. Supervisorの最終回答では、Claude CodeとCodexの一致点、違い、採用した見立てを分けて書く。

## 注意点

- `objective` にはユーザーが入力した調査目的だけを入れる。
- 手順、出力形式、プロンプト例を `objective` に混ぜない。
- 2ターンで終わると決め打ちしない。質問が続く場合は `turn` を増やして継続する。
