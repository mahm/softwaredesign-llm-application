// TypeSafeのSDKでJevを呼び出す最小の例
// 問い合わせ001の種別をchoiceの質問1つで判断させ、送るJSONと受け取るJSONを表示する
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";

// 問い合わせの本文をそのままstateとして渡す
const inquiry = await Bun.file(new URL("../inquiries/001.txt", import.meta.url)).text();

// リクエストの内容
const request = {
  model: "jev-latest",
  state: inquiry,
  questions: {
    inquiry_type: choice("この問い合わせの種別はどれですか", {
      トラブル: "これまで使えていたものが使えない、正しく動かない",
      使い方の質問: "操作方法、設定方法、社内ルールを知りたい",
      "申請・依頼": "アカウント、権限、機器、ソフトウェアの用意や変更を頼みたい",
      セキュリティの報告: "不審なメール、不正アクセスの疑い、端末の紛失などを知らせている",
      その他: "上のどれにも当てはまらない",
    }),
  },
};
console.log("リクエスト:");
console.log(JSON.stringify(request, null, 2));

// クライアントは環境変数TYPESAFE_API_KEYからAPIキーを読み込む
const client = new TypeSafeClient();
const response = await client.systemOne(request);
console.log("レスポンス:");
console.log(JSON.stringify(response, null, 2));
