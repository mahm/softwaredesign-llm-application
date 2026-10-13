// 30件の問い合わせを指定した回数ずつ分類し、正解ラベル（labels.tsv）と比べて集計する例
// 記事の実行結果（表5〜表11）に当たる値をタブ区切りで表示する。実行：bun samples/05-evaluate.ts [回数（省略時は10）]
import { TypeSafeClient } from "@typesafe-ai/sdk";
import { inquiryQuestions } from "./03-inquiry-questions.ts";

// 04-classify.tsと同じく、影響範囲と緊急度の回答の組み合わせから優先度を決める
const PRIORITY: Record<string, Record<string, string>> = {
  本人だけ: { 急がない: "低", 早めに対応: "低", すぐに対応: "中" },
  部署やチームの複数人: { 急がない: "低", 早めに対応: "中", すぐに対応: "高" },
  複数の部署や全社: { 急がない: "中", 早めに対応: "高", すぐに対応: "緊急" },
};

// 正解ラベルと比べる項目（labels.tsvの列名）と、表示に使う名前
const ITEMS: Record<string, string> = {
  inquiry_type: "種別",
  category: "対象",
  department: "担当部署",
  impact: "影響範囲",
  urgency: "緊急度",
  multiple_departments: "複数部署",
  needs_clarification: "要確認",
  priority: "優先度",
};

type Row = Record<string, string>;
type Run = { id: string; answers: Record<string, any>; ms: number; inputTokens: number };

const repeats = Number(process.argv[2] ?? 10);
const root = new URL("../", import.meta.url).pathname;

// タブ区切りのファイルを、1行目の列名をキーにしたオブジェクトの配列として読み込む
async function readTsv(path: string): Promise<Row[]> {
  const [header, ...rows] = (await Bun.file(path).text()).trim().split("\n").map((line) => line.split("\t"));
  return rows.map((row) => Object.fromEntries(header.map((key, i) => [key, row[i]])));
}

// 値ごとの件数を数える
function countBy(values: string[]) {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return counts;
}

// 正解ラベルを問い合わせの番号ごとにまとめる。用件が2つある問い合わせ（010、028）は2行ある
const labels = new Map<string, Row[]>();
for (const row of await readTsv(root + "labels.tsv")) labels.set(row.id, [...(labels.get(row.id) ?? []), row]);

// 30件を指定した回数ずつ分類し、回答と所要時間、入力トークン数を記録する
const client = new TypeSafeClient();
const inquiries = await readTsv(root + "inquiries/received-at.tsv");
const runs: Run[] = [];
for (let n = 1; n <= repeats; n++) {
  for (const { id, received_at } of inquiries) {
    const inquiry = await Bun.file(`${root}inquiries/${id}.txt`).text();
    const started = performance.now();
    const { answers, usage } = await client.systemOne({
      model: "jev-latest",
      state: { received_at, inquiry },
      questions: inquiryQuestions,
    });
    runs.push({ id, answers, ms: performance.now() - started, inputTokens: usage.input_tokens });
  }
  console.error(`${n}/${repeats}回目の分類が終わりました`);
}

// 回答を問い合わせの番号ごとにまとめる
const byInquiry = new Map<string, Run[]>();
for (const run of runs) byInquiry.set(run.id, [...(byInquiry.get(run.id) ?? []), run]);

// 回答を正解ラベルと同じ表記にする。noulは「はい」の確率がしきい値を超えたら「はい」とみなす
function valueOf(run: Run, item: string, threshold = 0.5): string {
  const { answers } = run;
  if (item === "priority") return PRIORITY[answers.impact.choice][answers.urgency.choice];
  const answer = answers[item];
  return answer.type === "noul" ? (answer.noul > threshold ? "はい" : "いいえ") : answer.choice;
}

// どちらかの用件の正解ラベルと一致すれば、正解ラベルどおりとする
function matches(run: Run, item: string, threshold = 0.5) {
  return labels.get(run.id)!.some((label) => label[item] === valueOf(run, item, threshold));
}

// 担当部署の振り分け：複数部署が「はい」か、担当部署のconfidenceが0.5未満なら人間が確認する
function routeOf(run: Run) {
  const toHuman = run.answers.multiple_departments.noul > 0.5 || run.answers.department.confidence < 0.5;
  // 人間による確認が必要なのは、正解ラベルで複数部署が「はい」の問い合わせと、正解ラベルと異なる担当部署を選んだ回
  const needed = labels.get(run.id)![0].multiple_departments === "はい" || !matches(run, "department");
  if (toHuman) return needed ? "人間が確認する対象になった（確認が必要）" : "人間が確認する対象になった（確認は不要）";
  return needed ? "正解ラベルと異なる担当部署へそのまま振り分けられた" : "正しい担当部署へそのまま振り分けられた";
}

const labelOf = (id: string, item: string) => [...new Set(labels.get(id)!.map((label) => label[item]))].join("、");
const range = (values: number[]) => `${Math.min(...values).toFixed(2)}〜${Math.max(...values).toFixed(2)}`;
const times = (counts: Map<string, number>) => [...counts].map(([value, n]) => `${value}（${n}回）`).join("、");
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const print = (...cells: (string | number)[]) => console.log(cells.join("\t"));

print("■ 質問ごとの回答と正解ラベルの一致（表5）");
print("質問", `${repeats}回とも正解ラベルどおり`, `${repeats}回とも正解ラベルと異なる`, "回によって異なる（異なった回数）");
for (const [item, name] of Object.entries(ITEMS)) {
  const same: string[] = [];
  const differ: string[] = [];
  const mixed: string[] = [];
  for (const [id, list] of byInquiry) {
    const wrong = list.filter((run) => !matches(run, item)).length;
    if (wrong === 0) same.push(id);
    else if (wrong === list.length) differ.push(id);
    else mixed.push(`${id}（${wrong}回）`);
  }
  print(name, `${same.length}件`, differ.join("、") || "なし", mixed.join("、") || "なし");
}

print("\n■ 問い合わせごとの振り分けの結果（表6）");
const routes = new Map<string, string[]>();
const routingErrors: string[] = [];
for (const [id, list] of byInquiry) {
  const counts = countBy(list.map(routeOf));
  const result = counts.size === 1
    ? `${repeats}回とも${[...counts.keys()][0]}`
    : `回によって異なった（${[...counts].map(([route, n]) => `${route}：${n}回`).join("、")}）`;
  routes.set(result, [...(routes.get(result) ?? []), id]);
  if ([...counts.keys()].some((route) => !route.startsWith("正しい") && !route.endsWith("（確認が必要）"))) routingErrors.push(id);
}
for (const [result, ids] of routes) print(result, `${ids.length}件`, ids.join("、"));

print("\n■ 振り分けに誤りが出た問い合わせの確率（表7。確率が0.1以上になった担当部署を表示）");
print("番号", "正解ラベルの担当部署", "担当部署の確率（最小〜最大）", "複数部署が「はい」の確率（最小〜最大）");
for (const id of routingErrors) {
  const list = byInquiry.get(id)!;
  const probabilities = list.map((run) => run.answers.department.probabilities as Record<string, number>);
  const departments = Object.keys(probabilities[0]).filter((d) => probabilities.some((p) => p[d] >= 0.1));
  print(
    id,
    labelOf(id, "department"),
    departments.map((d) => `${d} ${range(probabilities.map((p) => p[d]))}`).join("、"),
    range(list.map((run) => run.answers.multiple_departments.noul)),
  );
}

print("\n■ 優先度が正解ラベルと異なった問い合わせ（表8）");
print("番号", "正解ラベルの優先度", "回答から決めた優先度（回数）", "正解ラベルと異なった回答（回数）");
for (const [id, list] of byInquiry) {
  const wrong = list.filter((run) => !matches(run, "priority"));
  if (wrong.length === 0) continue;
  // 優先度が異なった回で、影響範囲と緊急度のどちらが正解ラベルと異なったかを数える
  const causes = wrong.flatMap((run) =>
    ["impact", "urgency"].filter((item) => !matches(run, item)).map((item) => `${ITEMS[item]}が「${valueOf(run, item)}」`),
  );
  print(id, labelOf(id, "priority"), times(countBy(list.map((run) => valueOf(run, "priority")))), times(countBy(causes)));
}

print("\n■ 要確認が正解ラベルと異なった問い合わせ（表9）");
print("番号", "正解ラベル", "「はい」の確率（最小〜最大）", "「はい」になった回数");
for (const [id, list] of byInquiry) {
  if (list.every((run) => matches(run, "needs_clarification"))) continue;
  const yes = list.map((run) => run.answers.needs_clarification.noul as number);
  print(id, labelOf(id, "needs_clarification"), range(yes), `${yes.filter((p) => p > 0.5).length}回`);
}

print("\n■ しきい値を変えたときに要確認が正解ラベルと異なった問い合わせ（表10）");
print("しきい値", "確認が必要なのに「いいえ」になった問い合わせ", "確認が不要なのに「はい」になった問い合わせ");
for (const threshold of [0.5, 0.45, 0.4]) {
  const missed = new Set<string>();
  const extra = new Set<string>();
  for (const run of runs) {
    if (matches(run, "needs_clarification", threshold)) continue;
    (valueOf(run, "needs_clarification", threshold) === "いいえ" ? missed : extra).add(run.id);
  }
  print(threshold, [...missed].sort().join("、") || "なし", [...extra].sort().join("、") || "なし");
}

print("\n■ TypeSafeのAPIを呼び出したときの所要時間と入力トークン数（表11）");
const ms = runs.map((run) => run.ms).sort((a, b) => a - b);
print("所要時間の中央値", `${Math.round(median(ms))}ミリ秒`);
print("所要時間の90パーセンタイル", `${Math.round(ms[Math.floor(ms.length * 0.9)])}ミリ秒`);
print("入力トークン数の中央値", Math.round(median(runs.map((run) => run.inputTokens))));
