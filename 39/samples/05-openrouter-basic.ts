// OpenRouterのDecisions APIで、01-basic-choice.tsと同じ種別の質問を送る例
// モデルIDは引数で指定する（省略時はtypesafe/jev-1.13）。レスポンス全体をJSONで表示する
import { OpenRouter } from "@openrouter/sdk";

const inquiry = await Bun.file(new URL("../inquiries/001.txt", import.meta.url)).text();

// APIキーは環境変数OPENROUTER_API_KEYから読み込んで渡す
const openrouter = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY ?? "" });
const response = await openrouter.alpha.decisions.create({
  decisionsRequest: {
    model: process.argv[2] ?? "typesafe/jev-1.13",
    state: inquiry,
    // TypeSafeのSDKのchoice関数は使わず、質問の内容（type、instructions、criteria）をそのまま書く
    questions: {
      inquiry_type: {
        type: "choice",
        instructions: "この問い合わせの種別はどれですか",
        criteria: {
          トラブル: "これまで使えていたものが使えない、正しく動かない",
          使い方の質問: "操作方法、設定方法、社内ルールを知りたい",
          "申請・依頼": "アカウント、権限、機器、ソフトウェアの用意や変更を頼みたい",
          セキュリティの報告: "不審なメール、不正アクセスの疑い、端末の紛失などを知らせている",
          その他: "上のどれにも当てはまらない",
        },
      },
    },
  },
});

console.log(JSON.stringify(response, null, 2));
