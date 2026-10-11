// OpenRouterのDecisions APIで、01-basic-choice.tsと同じ種別の質問を送る例
// モデルIDは引数で指定する（省略時はtypesafe/jev-1.13）。送るJSONと受け取るJSONを表示する
const inquiry = await Bun.file(new URL("../inquiries/001.txt", import.meta.url)).text();

// 質問は、TypeSafeのSDKのchoice関数が作るものと同じ形（type、instructions、criteria）で書く
const request = {
  model: process.argv[2] ?? "typesafe/jev-1.13",
  state: inquiry,
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
};
console.log("リクエスト:");
console.log(JSON.stringify(request, null, 2));

// APIキーは環境変数OPENROUTER_API_KEYから読み込み、Authorizationヘッダーで渡す
const response = await fetch("https://openrouter.ai/api/alpha/decisions", {
  method: "POST",
  headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify(request),
});
console.log("レスポンス:");
console.log(JSON.stringify(await response.json(), null, 2));
