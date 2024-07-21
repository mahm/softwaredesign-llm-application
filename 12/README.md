## サンプルコードの実行方法

以下のコマンドでサンプルコードが保存されているリポジトリをクローン後、

```
$ git clone https://github.com/mahm/softwaredesign-llm-application.git
```

続けて以下のコマンドを実行し、必要なライブラリのインストールを行って下さい。

```
$ cd softwaredesign-llm-application/12
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt # 必要なライブラリのインストール
```

次に環境変数の設定を行います。まず`.env.sample`ファイルをコピーして`.env`ファイルを作成します。

```
$ cp .env.sample .env
$ vi .env # お好きなエディタで編集してください
```

続けて.envファイル内の`OPENAI_API_KEY`と`TAVILY_API_KEY`を設定して下さい。`TAVILY_API_KEY`の取得方法については次節で解説します。

```
OPENAI_API_KEY=[発行されたAPIキーを設定します]
TAVILY_API_KEY=[発行されたAPIキーを設定します]
```

ディレクトリ内にはサンプルコードを収録した`arag_agent.py`が保存されています。お手元で動作を確認される際には、上記のセットアップの後に以下のコマンドを実行してご確認ください。`--task`オプションに続けて作成して欲しいレポートの指示を入力すると、レポートを生成します。

```
python arag_agent.py --task 生成AIスタートアップの最新動向について調査してください
```

## Tavily APIキーの取得方法

### Tavilyについて

Tavilyは、LLMアプリケーションにおけるRAG（Retrieval-Augmented Generation）に特化した検索エンジンです。

通常の検索APIはユーザーのクエリに基づいて検索結果を取得しますが、検索の目的に対して無関係な結果が返ってくることがあります。また、単純なURLやスニペットが返されるため、開発者が関連するコンテンツをスクレイピングしたり、不要な情報をフィルタリングしたりする必要があります。

一方でTavilyは、一回のAPI呼び出しで20以上のサイトを集約し、独自のAIを利用しタスクやクエリ、目的に最も関連する情報源とコンテンツを評価、フィルタリング、ランク付けすることが可能な仕組みになっています。

今回のサンプルコードを実行するためにはTavilyのAPIキーが必要なので、以下の手段でAPIキーを取得してください。

### Tavily APIキーの取得

1. まず https://tavily.com/ にアクセスし、画面右上の「Try it out」をクリックします。
2. するとサインイン画面に遷移するので、そのままサインインを進めて下さい。GoogleアカウントかGitHubアカウントを選択することができます。
3. サインインが完了するとAPIキーが取得できる画面に遷移します。画面中央部の「API Key」欄より、APIキーをコピーしてください。
4. APIキーを`.env`内の`TAVILY_API_KEY`に設定してください。

## LangSmith APIキーの取得方法

LangSmithはLLM（大規模言語モデル）実行のログ分析ツールです。LLMアプリケーションの実行を詳細に追跡し、デバッグや評価を行うための機能を提供します。詳細については https://docs.smith.langchain.com/ をご確認ください。

LangSmithによるトレースを有効にするためには、LangSmithにユーザー登録した上で、APIキーを発行する必要があります。以下の手順を参考にAPIキーを取得してください。

### LangSmith APIキーの取得

1. [サインアップページ](https://smith.langchain.com/)より、LangSmithのユーザー登録を行います。
2. [設定ページ](https://smith.langchain.com/settings)を開きます。
3. API keysタブを開き、「Create API Key」をクリックします。するとAPIキーが発行されますので、APIキーをコピーしてください。
4. APIキーを`.env`内の`LANGCHAIN_API_KEY`に設定してください。
5. また`LANGCHAIN_TRACING_V2`に`true`、`LANGCHAIN_PROJECT`に`sd-12`と設定してください。