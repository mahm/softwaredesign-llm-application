// choice、score、noulの3種類の質問を1回のリクエストで送る例
// 問い合わせ001について、種別、緊急度、要確認の3つを聞き、送るJSONと受け取るJSONを表示する
import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

const inquiry = await Bun.file(new URL("../inquiries/001.txt", import.meta.url)).text();

// 同じstateに対する質問は、1回のリクエストにまとめて送る
const request = {
  model: "jev-latest",
  // 緊急度は受付日時の時点で判断するように聞くため、受付日時と本文にそれぞれ名前を付けて渡す
  state: { received_at: "2026-10-19T09:12:00+09:00", inquiry },
  questions: {
    // 種別
    inquiry_type: choice("この問い合わせの種別はどれですか", {
      トラブル: "これまで使えていたものが使えない、正しく動かない",
      使い方の質問: "操作方法、設定方法、社内ルールを知りたい",
      "申請・依頼": "アカウント、権限、機器、ソフトウェアの用意や変更を頼みたい",
      セキュリティの報告: "不審なメール、不正アクセスの疑い、端末の紛失などを知らせている",
      その他: "上のどれにも当てはまらない",
    }),
    // 緊急度（0〜2の3段階）
    urgency: score("`received_at`の時点で、どれくらい急いで対応する必要がありますか", [
      "急がない（業務は続けられる。期限が書かれていないか、余裕がある）",
      "業務の一部に支障がある、または期限が近い",
      "業務が止まっている、または被害が広がるおそれがあり、すぐに対応が必要",
    ]),
    // 要確認：対応を始める前に、問い合わせた人への確認が必要か
    needs_clarification: noul({
      question: "対応を始める前に、問い合わせた人へ確認しなければならないことがありますか",
      focus: "申請・依頼なら対象者と期日、トラブルなら症状やエラーの内容が分かれば、対応を始められる",
    }, {
      true: "誰の、何についての依頼か、症状やエラーの内容などが分からず、確認しないと対応を始められない",
      false: "書かれている内容で対応を始められる。細かい点は対応しながら確認できる",
    }),
  },
};
console.log("リクエスト:");
console.log(JSON.stringify(request, null, 2));

const client = new TypeSafeClient();
const response = await client.systemOne(request);
console.log("レスポンス:");
console.log(JSON.stringify(response, null, 2));
