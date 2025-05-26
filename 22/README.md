# Software Design誌「実践LLMアプリケーション開発」第22回サンプルコード

## サンプルコードの実行方法

### プロジェクトのセットアップ

※ このプロジェクトは`uv`を使用しています。`uv`のインストール方法については[こちら](https://github.com/astral-sh/uv)をご確認ください。

以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ uv sync
```

次に環境変数の設定を行います。まず`.env.sample`ファイルをコピーして`.env`ファイルを作成します。

```
$ cp .env.sample .env
$ vi .env # お好きなエディタで編集してください
```

`.env`ファイルを編集し、以下のAPIキーを設定してください。

- `ANTHROPIC_API_KEY`: Claude APIのキー

### 実行方法

```bash
uv run run.py
```

## Streamlitアプリとエージェントとの通信の全体像

Streamlitアプリとエージェントとの通信の全体像は次の図の通りです。コード理解の際にお役立てください。

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant UI as Streamlit UI
    participant Agent as 領収書OCRエージェント
    participant Tasks as タスク (OCR/会計処理)
    
    note over UI: 初期状態: WorkflowState.INITIAL
    
    User->>UI: 画像アップロード
    User->>UI: 「処理開始」ボタンクリック
    
    note over UI: 状態変更: WorkflowState.PROCESSING
    
    UI->>Agent: receipt_workflow(image_path, thread_id=X)
    
    Agent->>Tasks: process_and_ocr_image()
    Tasks-->>Agent: OCR結果
    
    %% OCRイベントのストリーミング
    Agent-->>UI: StreamWriter: OCR_DONE イベント
    
    note over UI: UI更新: OCR結果を保存
    note over UI: st.session_state.ocr_text, st.session_state.ocr_result を更新
    
    Agent->>Tasks: generate_account_suggestion()
    Tasks-->>Agent: 勘定科目提案
    
    %% 提案イベントのストリーミング
    Agent-->>UI: StreamWriter: ACCOUNT_SUGGESTED イベント
    
    note over UI: UI更新: 会計情報を保存
    note over UI: st.session_state.account_info を更新
    
    %% interruptによる一時停止
    Agent-->>UI: interrupt() - 割り込み情報
    
    note over UI: 状態変更: WorkflowState.WAIT_FEEDBACK
    note over UI: st.rerun() で画面更新
    
    %% ユーザーフィードバックの待機状態
    UI-->>User: OCR結果と勘定科目提案を表示
    UI-->>User: フィードバック入力フォームを表示
    
    alt ユーザーが修正を要求
        User->>UI: フィードバック入力 + 「再生成」ボタンクリック
        
        note over UI: 状態変更: WorkflowState.PROCESSING
        
        UI->>Agent: Command(resume=Feedback(REGENERATE, text))
        
        %% タスクの再実行（OCRはキャッシュから）
        Agent->>Tasks: process_and_ocr_image() (キャッシュから)
        Tasks-->>Agent: キャッシュされたOCR結果
        
        Agent->>Tasks: generate_account_suggestion(feedback_history=[text])
        Tasks-->>Agent: 更新された勘定科目提案
        
        %% 再び割り込み
        Agent-->>UI: interrupt() - 更新された提案情報
        
        note over UI: 状態変更: WorkflowState.WAIT_FEEDBACK
        note over UI: UI更新: st.session_state.account_info を更新
        note over UI: st.rerun() で画面更新
        
        UI-->>User: 更新された勘定科目提案を表示
    else ユーザーが承認
        User->>UI: 「承認」ボタンクリック
        
        note over UI: 状態変更: WorkflowState.PROCESSING
        
        UI->>Agent: Command(resume=Feedback(APPROVE, ""))
        
        Agent->>Tasks: save_receipt_data()
        Tasks-->>Agent: 保存結果
        
        %% 保存完了イベント
        Agent-->>UI: StreamWriter: SAVE_COMPLETED イベント
        
        note over UI: 状態変更: WorkflowState.WORKFLOW_COMPLETED
        note over UI: st.rerun() で画面更新
        
        UI-->>User: 完了メッセージを表示
    end
```