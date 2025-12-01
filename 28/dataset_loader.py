"""
Dataset loader for file exploration agent training and evaluation - DIVERSE MULTI-DIRECTORY VERSION

This module provides criteria-based datasets for training and evaluating the file exploration agent.
All tasks require multi-file exploration and are evaluated using explicit criteria with LLM as a Judge.

VERSION 2.0 - Multi-directory diversification (Directories: 09, 12, 17, 18, 20, 22, 23, 24, 26, 27, 28)
- Training: 10 examples across 7 directories (12, 17, 20, 23, 24, 26, 27)
- Test: 5 examples across 4 directories (09, 18, 22, 28)
- Domain Coverage: 39.3% (11/28 directories)
"""

import dspy
from typing import List


def load_training_dataset() -> List[dspy.Example]:
    """
    Load diversified training dataset for file exploration tasks.

    10 high-quality training examples requiring multi-file exploration:
    - Easy: 3 tasks (2-3 files)
    - Medium: 4 tasks (3-5 files)
    - Hard: 3 tasks (5-8 files)

    Directories used: 12, 17, 20, 23, 24, 26, 27 (7 different directories)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    examples = [
        # ===== EASY 1: Dir 12 - Adaptive RAG (2 files) =====
        dspy.Example(
            task="arag_agent.pyで定義されている3つのRAG手法（メソッドA、B、C）を特定し、それぞれがどのノード関数にマッピングされているかを説明せよ",
            working_directory="../12",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - arag_agent.pyを読んだか: 2点

2. 必須要素の言及（6点）
   - メソッドA（"A"）の特定: 1点
   - メソッドB（"B"）の特定: 1点
   - メソッドC（"C"）の特定: 1点
   - メソッドAのマッピング先（"non_retrieval_qa"ノード）: 1点
   - メソッドBのマッピング先（"single_step_approach"ノード）: 1点
   - メソッドCのマッピング先（"multi_step_approach"ノード）: 1点

3. 情報統合（2点）
   - add_conditional_edgesでのメソッド分岐ロジックを説明: 1点
   - 各メソッドの実行後にENDへ遷移することに言及: 1点

オプション（加点）:
   - multi_step_approach.pyまたはsingle_step_approach.pyで実装詳細を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== EASY 2: Dir 17 - Task Planner (3 files) =====
        dspy.Example(
            task="my_agent/agent.pyのグラフ構造を調査し、task_planner → task_executor → reporter の3つのノードがどのようにエッジで接続されているかを説明せよ",
            working_directory="../17",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - my_agent/agent.pyを読んだか: 2点

2. 必須要素の言及（6点）
   - task_plannerノードの特定（create_task_planner_agent関数の呼び出し）: 1.5点
   - task_executorノードの特定（create_task_executor_agent関数の呼び出し）: 1.5点
   - reporterノードの特定（Reporterクラスのインスタンス）: 1点
   - エッジ接続: START → task_planner: 0.5点
   - エッジ接続: task_planner → task_executor: 0.75点
   - エッジ接続: task_executor → reporter → END: 0.75点

3. 情報統合（2点）
   - グラフの線形フロー（3ノードの順次実行）を説明: 1点
   - MemorySaverによるチェックポイント機能に言及: 1点

オプション（加点）:
   - task_planner_agent.pyまたはtask_executor_agent.pyでサブグラフ実装を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== EASY 3: Dir 26 - DSPy RAG MIPROv2 (2 files) =====
        dspy.Example(
            task="rag_module.pyの3つのクラス（RewriteQuery、GenerateAnswer、RAGQA）を特定し、RAGQAクラスのforwardメソッドで検索実行に使用されているコード（dspy.settings.rm呼び出し）を説明せよ",
            working_directory="../26",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（1点）
   - rag_module.pyを読んだか: 1点

2. 必須要素の言及（7点）
   - RewriteQueryクラスの特定（dspy.Signature継承）: 1.5点
   - GenerateAnswerクラスの特定（dspy.Signature継承）: 1.5点
   - RAGQAクラスの特定（dspy.Module継承）: 1点
   - forwardメソッドでのdspy.settings.rm(rewritten)呼び出し: 2点
   - result.passagesでのパッセージ取得: 1点

3. 情報統合（2点）
   - RAGQAのforwardメソッドの処理フロー（rewrite → retrieve → generate）を説明: 1.5点
   - dspy.Retrieveを使わずに直接rmを呼び出す理由に言及できれば: 0.5点

オプション（加点）:
   - CLAUDE.mdまたはREADME.mdでdspy.Retrieve非使用の理由を確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== MEDIUM 1: Dir 20 - MCP + LangGraph (3 files) =====
        dspy.Example(
            task="src/sd_20/mcp_manager.pyのMCPツールロード処理を調査し、create_langchain_tool関数がMCPツールをLangChainのStructuredToolに変換する仕組みを説明せよ。また、src/sd_20/agent.pyでのツール使用箇所を特定せよ",
            working_directory="../20",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - src/sd_20/mcp_manager.pyを読んだか: 1点
   - src/sd_20/agent.pyを読んだか: 1点

2. 必須要素の言及（6点）
   - create_langchain_tool関数の特定: 1点
   - MCPToolからStructuredToolへの変換処理: 1.5点
   - full_tool_name（prefixとtool_nameの結合）の生成: 1点
   - load_all_mcp_tools関数でのツールロード: 1点
   - agent.pyでのtools = asyncio.run(load_all_mcp_tools())呼び出し: 1点
   - create_react_agentへのtools引数渡し: 0.5点

3. 情報統合（2点）
   - MCPサーバー → mcp_manager → agent の連携フローを説明: 1点
   - プレフィックスによるツール名の重複防止戦略に言及: 1点

オプション（加点）:
   - src/mcp_servers/server.pyでMCPツール定義を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== MEDIUM 2: Dir 23 - Multi-agent Patterns (4 files) =====
        dspy.Example(
            task="src/sd_23/supervisor_graph.pyとsrc/sd_23/swarm_graph.pyを比較し、2つのマルチエージェントパターン（SupervisorとSwarm）の違いを説明せよ。また、各パターンで使用されているエージェントを特定せよ",
            working_directory="../23",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - src/sd_23/supervisor_graph.pyを読んだか: 1点
   - src/sd_23/swarm_graph.pyを読んだか: 1点
   - src/sd_23/agents/内の少なくとも1つのエージェントファイルを読んだか: 1点

2. 必須要素の言及（5点）
   - Supervisorパターンの特定（create_supervisor関数使用）: 1点
   - Supervisorで使用されるエージェント（math_agent、research_agent）: 1点
   - Swarmパターンの特定（create_swarm関数使用）: 1点
   - Swarmで使用されるエージェント（faq_agent、tech_agent）: 1点
   - default_active_agent="faq_support"の指定: 0.5点
   - supervisor_promptまたはsupervisor_modelの設定: 0.5点

3. 情報統合（2点）
   - SupervisorとSwarmの調整メカニズムの違いを説明: 1.5点
   - 各パターンの用途の違い（タスク委譲 vs エージェント切り替え）に言及: 0.5点

オプション（加点）:
   - README.mdで2つのパターンの説明を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== MEDIUM 3: Dir 24 - Hierarchical Task Agent (4 files) =====
        dspy.Example(
            task="src/sd_24/utils/todo_manager.pyのTodoManagerクラスを調査し、タスク追加（add_task）、ステータス更新（update_status）、未完了タスク取得（get_pending_tasks）の3つのメソッドの実装を説明せよ",
            working_directory="../24",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（2点）
   - src/sd_24/utils/todo_manager.pyを読んだか: 2点

2. 必須要素の言及（6点）
   - add_taskメソッドの特定（task_id生成、TodoItem作成）: 1.5点
   - タスクID生成ロジック（f"TASK-{self.task_counter:04d}"）: 1点
   - update_statusメソッドの特定（status更新、updated_at更新）: 1.5点
   - get_pending_tasksメソッドの特定（TaskStatus.PENDINGフィルタリング）: 1.5点
   - エージェント別フィルタリング機能（agent引数）: 0.5点

3. 情報統合（2点）
   - TodoItemデータモデル（Pydantic BaseModel）の構造を説明: 1点
   - TodoManagerのシングルトンパターン（todo_manager = TodoManager()）に言及: 1点

オプション（加点）:
   - src/sd_24/main.pyまたはagents/内でのtodo_manager使用箇所を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== MEDIUM 4: Dir 26 - DSPy RAG MIPROv2 (4 files) =====
        dspy.Example(
            task="rag_optimization.pyのMIPROv2最適化プロセスを調査せよ。SMART_MODELとFAST_MODELの役割分担、optimizer.compile呼び出しの引数（trainset、valset、minibatch）、最適化前後の評価方法を説明せよ",
            working_directory="../26",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - rag_optimization.pyを読んだか: 1.5点
   - config.pyを読んだか: 1点
   - rag_module.pyまたはevaluator.pyを読んだか: 0.5点

2. 必須要素の言及（5点）
   - SMART_MODELの役割（prompt_model、temperature=0.0）: 1点
   - FAST_MODELの役割（推論用、dspy.configure(lm=fast_lm)）: 1点
   - dspy.MIPROv2初期化（metric=rag_comprehensive_metric、auto="medium"）: 1点
   - optimizer.compile引数（trainset、valset、minibatch=True）: 1.5点
   - 評価関数（evaluation関数でベースラインと最適化版を比較）: 0.5点

3. 情報統合（2点）
   - train/val分割（50:50）とtest評価の戦略を説明: 1点
   - 最適化フロー（ベースライン評価 → MIPROv2最適化 → 最適化後評価）を説明: 1点

オプション（加点）:
   - README.mdまたはCLAUDE.mdでMIPROv2の効果（数値）を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== HARD 1: Dir 24 - Hierarchical Task Agent (7 files) =====
        dspy.Example(
            task="src/sd_24/の階層型タスクエージェントの全体アーキテクチャを調査せよ。main.py → agents/task_decomposer.py → utils/todo_manager.py → agents/writer.py の連携フローと、各コンポーネントの役割を説明せよ",
            working_directory="../24",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（4点）
   - src/sd_24/main.pyを読んだか: 1点
   - src/sd_24/agents/task_decomposer.pyを読んだか: 1点
   - src/sd_24/utils/todo_manager.pyを読んだか: 1点
   - src/sd_24/agents/writer.pyまたはresearch.pyを読んだか: 1点

2. 必須要素の言及（4点）
   - main.pyでのワークフロー初期化（create_writing_assistant_workflow）: 0.5点
   - task_decomposerの役割（タスク分解、TODO生成）: 1点
   - todo_managerの役割（タスク管理、ステータス追跡）: 1点
   - writerまたはresearchエージェントの役割（タスク実行）: 1点
   - Supervisorパターンによる調整（create_supervisor）: 0.5点

3. 情報統合（2点）
   - 階層的計画フロー（タスク分解 → TODO登録 → エージェント実行 → ステータス更新）を説明: 1.5点
   - utils/ディレクトリの支援機能（memory.py、search_tools.py）に言及: 0.5点

オプション（加点）:
   - display/terminal_ui.pyまたはprogress_tracker.pyでUI実装を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== HARD 2: Dir 27 - DSPy RAG GEPA (6 files) =====
        dspy.Example(
            task="27ディレクトリのGEPA最適化システムの完全なデータフローを調査せよ。dataset_loader.py → rag_module.py → evaluator.py → rag_optimization_gepa.py → モデル保存 の全プロセスを説明せよ",
            working_directory="../27",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（3点）
   - dataset_loader.pyを読んだか: 0.5点
   - rag_module.pyを読んだか: 0.5点
   - evaluator.pyを読んだか: 0.5点
   - rag_optimization_gepa.pyを読んだか: 1点
   - config.pyまたはembeddings_cache.pyを読んだか: 0.5点

2. 必須要素の言及（5点）
   - load_jqara_dataset関数によるデータ読み込み（positive/negative分離）: 0.5点
   - dspy.Example形式への変換（.with_inputs）: 0.5点
   - RAGQAクラスのforwardメソッド（rewrite → retrieve → generate）: 1点
   - gepa_metric_with_feedback関数（dspy.Prediction with score and feedback）: 1点
   - dspy.GEPA初期化（reflection_lm、candidate_selection_strategy="pareto"）: 1点
   - optimizer.compile実行とモデル保存（タイムスタンプ、スコア、symlink）: 1点

3. 情報統合（2点）
   - データフロー全体（JSON → Example → RAG処理 → GEPA最適化 → 評価 → 保存）を説明: 1.5点
   - SMART_MODEL（reflection_lm、temperature=1.0）とFAST_MODEL（推論、temperature=0.0）の役割分担を説明: 0.5点

オプション（加点）:
   - README.mdでGEPA最適化結果（20%改善、35x効率化）を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== HARD 3: Dir 23 - Multi-agent Patterns (8 files) =====
        dspy.Example(
            task="src/sd_23/の全体像を調査せよ。supervisor_graph.pyとswarm_graph.pyの2つのパターンを比較し、各パターンで使用される全エージェント（math、research、faq、tech）の役割と、パターンごとのエージェント選択メカニズムを説明せよ",
            working_directory="../23",
            criteria="""以下の基準で0-10点で評価してください:

1. 必須ファイルの読み取り（4点）
   - src/sd_23/supervisor_graph.pyを読んだか: 1点
   - src/sd_23/swarm_graph.pyを読んだか: 1点
   - src/sd_23/agents/math_agent.pyを読んだか: 0.5点
   - src/sd_23/agents/research_agent.pyを読んだか: 0.5点
   - src/sd_23/agents/faq_agent.pyを読んだか: 0.5点
   - src/sd_23/agents/tech_agent.pyを読んだか: 0.5点

2. 必須要素の言及（4点）
   - Supervisorパターンのエージェント（math_agent、research_agent）: 0.5点
   - Swarmパターンのエージェント（faq_agent、tech_agent）: 0.5点
   - Supervisor調整メカニズム（supervisor_model、supervisor_prompt）: 1点
   - Swarm調整メカニズム（default_active_agent、エージェント切り替え）: 1点
   - 各エージェントの専門性（計算、リサーチ、FAQ、技術サポート）: 1点

3. 情報統合（2点）
   - 2つのパターンの違い（中央集権的タスク委譲 vs 分散的エージェント切り替え）を説明: 1.5点
   - パターンごとの用途（Supervisor: 複雑なタスク分解、Swarm: 動的な役割切り替え）に言及: 0.5点

オプション（加点）:
   - README.mdで2つのパターンの具体的使用例を確認: +0.5点
   - __init__.pyでのエージェントexport構造を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),
    ]

    return examples


def load_test_dataset() -> List[dspy.Example]:
    """
    Load diversified test dataset for file exploration tasks (DIFFERENT from training directories).

    5 high-quality test examples with unseen exploration patterns:
    - Easy: 1 task (1-2 files)
    - Medium: 2 tasks (2-3 files)
    - Hard: 2 tasks (4-8 files)

    Directories used: 09, 18, 22, 28 (4 different directories, ALL DIFFERENT from training set)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    examples = [
        # ===== Test Example 1: EASY - Dir 09 (1 file) =====
        dspy.Example(
            task="research_agent.pyのTaskデータクラスを特定し、タスクが持つ4つのフィールド（id、action、description、related_ids）の役割を説明せよ",
            working_directory="../09",
            criteria="""以下の基準で0-10点で評価してください:

【重要】research_agent.pyを読んでいない場合、総合スコアは0点です。

1. 必須ファイルの読み取り（1点）
   - research_agent.pyを読んだか: 1点
   - 読んでいない場合: 0点（総合スコアも0点）

2. 必須要素の言及（7点）
   - Taskクラスの特定（class Task(BaseModel)）: 1点
   - idフィールド（int型、タスク識別子）: 1.5点
   - actionフィールド（str型、タスクアクション種別）: 1.5点
   - descriptionフィールド（str型、タスク説明）: 1.5点
   - related_idsフィールド（list[int]、依存タスクID）: 1.5点

3. 情報統合（2点）
   - related_idsがタスク実行順序の依存関係を表現することに言及: 1点
   - BaseModelの継承（Pydantic使用）に言及: 1点

オプション（加点）:
   - find_next_task関数でrelated_idsがどのように使用されるか確認: +0.5点""",
            difficulty="easy"
        ).with_inputs("task", "working_directory"),

        # ===== Test Example 2: MEDIUM - Dir 18 (2 files) =====
        dspy.Example(
            task="sd_18/agent.pyのチャート生成機能を調査せよ。日本語フォント設定の実装（matplotlibのフォント設定）と、チャート保存パスの命名規則（timestampディレクトリの使用）を説明せよ",
            working_directory="../18",
            criteria="""以下の基準で0-10点で評価してください:

【重要】sd_18/agent.pyを読んでいない場合、総合スコアは0点です。
推測や一般的知識に基づく説明は一切認めません。ファイルを必ず読んでください。

1. 必須ファイルの読み取り（1点）
   - sd_18/agent.pyを読んだか: 1点
   - 読んでいない場合: 0点（総合スコアも0点）

2. 必須要素の言及（7点）
   - matplotlib.rc('font', family=...)による日本語フォント設定: 2点
   - フォント候補リスト（'Hiragino Sans', 'MS Gothic', 'Yu Gothic'など）: 1.5点
   - output/{timestamp}/ディレクトリへのチャート保存: 1.5点
   - チャートファイル名にタイムスタンプまたは連番が付与されること: 1点
   - python_repl_toolでのmatplotlib.pyplot使用: 1点
   ※ファイルを読まずに推測で説明した場合は全て0点

3. 情報統合（2点）
   - 日本語フォント設定が日本語ラベルの文字化け防止のために必要な理由: 1点
   - タイムスタンプ付きディレクトリによる実行履歴管理の利点: 1点

オプション（加点）:
   - FileReaderクラスでのチャート画像読み込み機能を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Test Example 3: MEDIUM - Dir 22 (3 files) =====
        dspy.Example(
            task="src/receipt_processor/vision.pyとmodels.pyとconstants.pyを調査し、Vision APIによるOCR処理の実装を説明せよ。build_vision_message関数、ReceiptOCRResultモデル、CLAUDE_FAST_MODEL定数の役割を特定せよ",
            working_directory="../22",
            criteria="""以下の基準で0-10点で評価してください:

【重要】
- 3ファイル全く読んでいない場合（0ファイル）: 総合スコア0点
- 一部のファイルを読んでいる場合: 読んだファイル数に応じて部分点
  例: 2ファイル読んだ場合 = 2/3 × 3点 = 2点
- ファイルが見つからない場合は、recursive=Trueとpattern='*.py'で再帰探索してください
- 推測や一般知識のみの説明は評価しない（ファイル内容に基づく説明のみ評価）

【ファイル読み取り証拠の判定方法】
reportとtrajectoryの両方を確認してください：

1. trajectory確認（優先）:
   - tool_name_N="read_file" かつ tool_args_N に対象ファイル名
   - → 確実に読んでいる ✓

2. report確認（補助）:
   - ファイル内の具体的な値の引用（例: CLAUDE_FAST_MODEL = "claude-3-5-haiku-20241022"）
   - "vision.pyを読んだ結果、..."などの明示的表現
   - コードの一部を示す
   - → 読んだ証拠あり ✓

3. 推測判定:
   - "おそらく", "一般的に", "通常は"
   - trajectoryに該当ツール呼び出しなし
   - 具体的な値の欠如
   - → 推測（ハルシネーション） ✗

trajectoryで読んでいることが確認できれば、reportの記述が不十分でもファイル読み取り点は付与してください。
ただし、reportの品質が低い場合はimprovement_suggestionsで「具体的な値を引用すべき」と指摘してください。

1. 必須ファイルの読み取り（3点）
   - vision.pyを読んだか: 1点
   - models.pyを読んだか: 1点
   - constants.pyを読んだか: 1点
   - 読んだファイル数に応じて比例配分

2. 必須要素の言及（5点）
   - build_vision_message関数の特定（画像のbase64エンコード処理）: 1.5点
   - ReceiptOCRResultモデルの特定（raw_text、date、amount、shop_name、itemsなどのフィールド）: 1.5点
   - CLAUDE_FAST_MODELの定数値（"claude-3-5-haiku-20241022"など）: 1点
   - システムプロンプトのOCRタスク指示内容: 0.5点
   - ChatAnthropicでのVision API呼び出し: 0.5点
   ※ファイルを読まずに推測で説明した場合は全て0点

3. 情報統合（2点）
   - 画像 → base64エンコード → Vision API → 構造化データ の流れを説明: 1点
   - CLAUDE_FAST_MODELがOCR処理に最適化されたモデルであることに言及: 1点

オプション（加点）:
   - agent.pyでocr_receipt関数の使用箇所を確認: +0.5点""",
            difficulty="medium"
        ).with_inputs("task", "working_directory"),

        # ===== Test Example 4: HARD - Dir 22 (5 files) =====
        dspy.Example(
            task="src/receipt_processor/のワークフロー全体を調査せよ。agent.py、vision.py、account.py、storage.py、models.pyの5ファイルを読み、画像アップロード → OCR処理 → 会計提案 → フィードバック受付 → CSV保存 の流れを説明せよ",
            working_directory="../22",
            criteria="""以下の基準で0-10点で評価してください:

【重要】
- 5ファイル全く読んでいない場合（0ファイル）: 総合スコア0点
- 一部のファイルを読んでいる場合: 読んだファイル数に応じて部分点
  例: 4ファイル読んだ場合 = 4/5 × 2.5点 = 2.0点
- ファイルが見つからない場合は、recursive=Trueとpattern='*.py'で再帰探索してください

1. 必須ファイルの読み取り（3点）
   - agent.pyを読んだか: 0.5点
   - vision.pyを読んだか: 0.5点
   - account.pyを読んだか: 0.5点
   - storage.pyを読んだか: 0.5点
   - models.pyを読んだか: 0.5点
   - 読んだファイル数に応じて比例配分（例: 4/5ファイル = 2.0点）
   - （ui_components.pyまたはapp.pyを読んだ場合は追加で+0.5点）

2. 必須要素の言及（5点）
   - process_and_ocr_image関数（@taskデコレータ、OCR処理）: 1点
   - ocr_receipt関数（vision.py、Vision API呼び出し）: 0.5点
   - generate_account_suggestion関数（@taskデコレータ、会計提案生成）: 1点
   - suggest_account_info関数（account.py、勘定科目提案ロジック）: 0.5点
   - save_to_csv関数（storage.py、CSV保存）: 0.5点
   - interrupt()による人間フィードバック待機: 1点
   - Feedbackモデル（CommandType: APPROVE/REGENERATE）: 0.5点

3. 情報統合（2点）
   - ワークフロー全体（画像 → OCR → 会計提案 → フィードバック → 保存）を説明: 1点
   - @taskとinterruptによるHuman-in-the-loopパターンの実装を説明: 1点

オプション（加点）:
   - README.mdでLangGraph @task機能の背景を確認: +0.5点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),

        # ===== Test Example 5: HARD - Dir 28 (4 files) =====
        dspy.Example(
            task="28ディレクトリのツール仕様保持メカニズムを調査せよ。agent_tool_specs.py、agent_module.py、agent_optimization_gepa.py、dataset_loader.pyの4ファイルを読み、tool_spec InputFieldがGEPA最適化でどのように使用され、なぜ重要かを説明せよ",
            working_directory="../28",
            criteria="""以下の基準で0-10点で評価してください:

【重要】
- 4ファイル全く読んでいない場合（0ファイル）: 総合スコア0点
- 一部のファイルを読んでいる場合: 読んだファイル数に応じて部分点
  例: 3ファイル読んだ場合 = 2.5点（agent_module.py 1点 + agent_optimization_gepa.py 1点 + agent_tool_specs.py 0.5点）

1. 必須ファイルの読み取り（3点）
   - agent_tool_specs.pyを読んだか: 0.5点
   - agent_module.pyを読んだか: 1点
   - agent_optimization_gepa.pyを読んだか: 1点
   - dataset_loader.pyを読んだか: 0.5点
   - 読んだファイル分の点数を合計

2. 必須要素の言及（5点）
   - generate_tool_specifications関数（agent_tool_specs.py、dspy.Tool使用）: 1点
   - FileExplorationSignatureのtool_spec InputField: 1.5点
   - agent_module.pyでのtool_spec生成（generate_tool_specifications呼び出し）: 1点
   - ls_directory、read_file、write_fileの3つのツール関数: 1点
   - tool_specがGEPA最適化対象外（InputFieldは最適化されない）: 0.5点

3. 情報統合（2点）
   - ツール仕様（名前・型・デフォルト値）が削除される問題とtool_spec InputFieldによる解決: 1点
   - tool_specがinstructionsとは独立して保持される仕組みを説明: 1点

オプション（加点）:
   - CLAUDE.mdでツール仕様保持の重要性（第1回失敗の教訓）を確認: +1点""",
            difficulty="hard"
        ).with_inputs("task", "working_directory"),
    ]

    return examples


def load_mini_test_dataset() -> List[dspy.Example]:
    """
    Load mini test dataset for quick GEPA optimization verification.

    3 representative examples from training dataset:
    - Easy: 1 task (Dir 12)
    - Medium: 1 task (Dir 20)
    - Hard: 1 task (Dir 24)

    Returns:
        List of dspy.Example instances with task, working_directory, and criteria fields
    """
    # Get full training dataset
    full_train = load_training_dataset()

    # Select representative examples: Task 1 (Easy), Task 4 (Medium), Task 7 (Hard)
    mini_examples = [
        full_train[0],  # Task 1 (Easy): Dir 12 - Adaptive RAG method mapping
        full_train[3],  # Task 4 (Medium): Dir 20 - MCP tool loading
        full_train[7],  # Task 8 (Hard): Dir 24 - Hierarchical agent architecture
    ]

    return mini_examples


def load_file_exploration_dataset(
    dataset_type: str = "train",
    random_seed: int = 42
) -> List[dspy.Example]:
    """
    Load file exploration dataset (train, test, or mini_test).

    Args:
        dataset_type: "train" (10 examples), "test" (5 examples), or "mini_test" (3 examples)
        random_seed: Random seed for reproducibility (currently unused, for future expansion)

    Returns:
        List of dspy.Example instances
    """
    if dataset_type == "train":
        examples = load_training_dataset()
        print(f"📚 File Exploration Training Dataset (DIVERSE) loaded:")
        print(f"  Training examples: {len(examples)}")
        print(f"  Easy: 3, Medium: 4, Hard: 3")
        print(f"  Directories: 12, 17, 20, 23, 24, 26, 27 (7 directories)")
        print(f"  Domain coverage: 25% (7/28 directories)")
    elif dataset_type == "test":
        examples = load_test_dataset()
        print(f"🧪 File Exploration Test Dataset (DIVERSE) loaded:")
        print(f"  Test examples: {len(examples)}")
        print(f"  Easy: 1, Medium: 2, Hard: 2")
        print(f"  Directories: 09, 18, 22, 28 (4 directories)")
        print(f"  ⚠️  ALL test directories are DIFFERENT from training set")
    elif dataset_type == "mini_test":
        examples = load_mini_test_dataset()
        print(f"⚡ File Exploration Mini Test Dataset (DIVERSE) loaded:")
        print(f"  Mini test examples: {len(examples)}")
        print(f"  Easy: 1, Medium: 1, Hard: 1")
        print(f"  Directories: 12, 20, 24 (3 directories)")
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Use 'train', 'test', or 'mini_test'.")

    return examples


def create_custom_example(
    task: str,
    working_directory: str = ".",
    criteria: str = "",
    difficulty: str = "medium"
) -> dspy.Example:
    """
    Create a custom file exploration example.

    Args:
        task: Task description
        working_directory: Working directory path
        criteria: Evaluation criteria (detailed scoring rubric)
        difficulty: Task difficulty ("easy", "medium", "hard")

    Returns:
        dspy.Example instance
    """
    return dspy.Example(
        task=task,
        working_directory=working_directory,
        criteria=criteria,
        difficulty=difficulty
    ).with_inputs("task", "working_directory")
