// 30件の問い合わせをJevで分類し、担当部署と優先度を決める例
// 1件を1行とし、番号、種別、対象、担当部署、優先度、複数部署、要確認をタブ区切りで表示する
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { inquiryQuestions } from "./03-inquiry-questions.ts";

// 影響範囲と緊急度の回答の組み合わせから優先度を決める表。表の内容は運用方針の一例
const PRIORITY = {
  本人だけ: { 急がない: "低", 早めに対応: "低", すぐに対応: "中" },
  部署やチームの複数人: { 急がない: "低", 早めに対応: "中", すぐに対応: "高" },
  複数の部署や全社: { 急がない: "中", 早めに対応: "高", すぐに対応: "緊急" },
};

const client = new TypeSafeClient();
const dir = new URL("../inquiries/", import.meta.url).pathname;
// 受付日時の一覧（番号と日時のタブ区切り）を読み込み、その順に問い合わせを処理する
const rows = (await Bun.file(dir + "received-at.tsv").text()).trim().split("\n").slice(1);

for (const [id, receivedAt] of rows.map((row) => row.split("\t"))) {
  const { answers } = await client.systemOne({
    model: "jev-latest",
    state: { received_at: receivedAt, inquiry: await Bun.file(`${dir}${id}.txt`).text() },
    questions: inquiryQuestions,
  });
  const priority = PRIORITY[answers.impact.choice][answers.urgency.choice];
  // noulは「はい」の確率なので、0.5を超えたら「はい」とみなす
  console.log([
    id, answers.inquiry_type.choice, answers.category.choice, answers.department.choice, priority,
    answers.multiple_departments.noul > 0.5 ? "複数部署" : "-",
    answers.needs_clarification.noul > 0.5 ? "要確認" : "-",
  ].join("\t"));
}
