// 問い合わせの緊急度と、対応前の確認の要否を1回のリクエストで尋ねる
import { readFile } from "node:fs/promises";
import { noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

const inquiry = await readFile(new URL("../inquiries/sdk-example.txt", import.meta.url), "utf8");

const request = {
  model: "jev-1.14.0",
  // 緊急度は受付日時の時点で判断したいので受付日時と本文にそれぞれ名前を付ける
  state: { received_at: "2026-10-19T09:12:00+09:00", inquiry },
  questions: {
    // 緊急度（0〜2の3段階）
    urgency: score("`received_at`の時点で、どれくらい急いで対応する必要がありますか", [
      "急がない（業務は続けられる。期限が書かれていないか、余裕がある）",
      "業務の一部に支障がある、または期限が近い",
      "業務が止まっている、または被害が広がるおそれがあり、すぐに対応が必要",
    ]),
    // 要確認：対応を始める前に問い合わせた人への確認が必要か
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

// クライアントは環境変数TYPESAFE_API_KEYからAPIキーを読み込む
const client = new TypeSafeClient();
const response = await client.systemOne(request);
console.log("レスポンス:");
console.log(JSON.stringify(response, null, 2));
