# 「実践LLMアプリケーション開発」サンプルコード

このリポジトリは技術評論社Software Design誌の連載「実践LLMアプリケーション開発」に関連するサンプルコードを保存しています。

## 注意事項

- このリポジトリのコードは自由に閲覧・二次利用が可能です。
- ただし、すべて自己責任での利用となります。開発者や連載記事の著者は、いかなる責任も負いません。

ソースコードに関する具体的な説明や解説については、連載記事の内容をご参照ください。

## サンプルコード一覧

| 回 | テーマ | 説明 |
|:---:|:---|:---|
| 01 | Chainlitを使った基本的なチャットボット | OpenAI APIとChainlitを使用したシンプルなチャットボットの実装 |
| 02 | RAGチャットボットの基礎 | ChromaDBとOpenAI Embeddingsを使用した「枝豆の妖精」スタイルのRAGチャットボット |
| 03 | LangChainによるRAG実装 | LangChainのRetrievalQAWithSourcesChainを使ったベクトルストア検索の実装 |
| 04 | Function Callingによるエージェント | LangChainとOpenAI Function Callingを使ったファイル管理エージェントの実装 |
| 05 | エージェントのフォールバック処理 | Function Calling失敗時のフォールバック処理とメモリ管理の実装 |
| 06 | Python環境構築 | 基本的なPython開発環境のセットアップ方法 |
| 07 | LangGraphによるユーザーインタビューグラフ | LangGraphを使った対話フローの実装 |
| 08 | LangGraphの応用 | LangGraphを使った複雑なワークフローの実装 |
| 09 | Research Agent（タスク分解型エージェント） | Tavily検索APIとLangGraphを使った調査レポート生成エージェント |
| 10 | CRAG（Corrective RAG） | Cross-Encoderを使った検索結果の再ランキングとCRAGパターンの実装 |
| 11 | ReAct Agent | create_react_agentを使った汎用的なリサーチエージェントの実装 |
| 12 | ARAG（Adaptive RAG） | タスクに応じて検索戦略を動的に切り替えるAdaptive RAGの実装とLangSmith連携 |
| 14 | Streamlitアプリケーション | Streamlitを使ったWebアプリケーションとLangGraphの連携 |
| 16 | LangGraph Platform | LangGraph CLIとLangGraph Platformによるエージェントのデプロイ |
| 17 | LangGraph Studio | LangGraph Studioを使ったエージェントの開発とデバッグ |
| 18 | データ分析エージェント | Claudeを使ったチャート生成とデータセット作成エージェント（日本語フォント対応） |
| 19 | MCP（Model Context Protocol）Server | MCPサーバーの実装とCursorエディタへの組み込み方法 |
| 20 | MCPとLangGraphエージェントの連携 | create_react_agentとMCPサーバーの統合、SQLite操作の実装 |
| 21 | LangGraphによるContent Creator | StreamlitとLangGraphを使ったX投稿作成エージェント、Human-in-the-Loopでのフィードバック改善 |
| 22 | Streamlitと領収書OCRエージェント | Claude Vision APIを使った領収書OCRと会計処理ワークフロー、Human-in-the-Loop実装 |
| 23 | SupervisorパターンとSwarmパターン | LangGraphのマルチエージェントパターン（Supervisor/Swarm）の実装と比較 |
| 24 | LangGraphエージェントの実践 | TavilyとClaudeを使った対話型調査エージェント、デバッグモード実装 |
| 25 | DSPyとMIPROv2によるチャットボット最適化 | MIPROv2を使った枝豆の妖精スタイルチャットボットのプロンプト自動最適化とMLflow連携 |
| 26 | DSPy MIPROv2によるRAG最適化 | JQaRAデータセットを使ったRAGパイプラインのMIPROv2最適化（検索クエリリライト+回答生成） |
| 27 | DSPy GEPAによるRAG最適化 | GEPAを使ったRAGパイプラインの最適化、Reflective feedbackによる改善 |
| 28 | DSPy GEPAによるReActエージェント最適化 | ファイル探索エージェントのGEPA自動最適化、LLM as a Judge評価、Tool仕様保持の実装 |
