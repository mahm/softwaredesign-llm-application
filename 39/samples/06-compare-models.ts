// OpenRouterのDecisions APIで、指定したモデルに30件の問い合わせを分類させる例
// 質問ごとの正解ラベルとの一致数、回答が変わった問い合わせの数、所要時間、利用料金を表示する
import { OpenRouter } from "@openrouter/sdk";
import { inquiryQuestions } from "./03-inquiry-questions.ts";

// 使い方：bun samples/06-compare-models.ts <モデルID> [繰り返し回数]
const model = process.argv[2] ?? "typesafe/jev-1.13";
const repeats = Number(process.argv[3] ?? 1);
const openrouter = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY ?? "" });
const readTsv = async (path: string) =>
  (await Bun.file(new URL(path, import.meta.url)).text()).trim().split("\n").map((row) => row.split("\t"));

// 正解ラベルを読み込む。列名は質問のキーと同じ
// 複数の担当部署にまたがる問い合わせは用件ごとに行があるため、いずれかの行と一致すれば正解とする
const [header, ...labels] = await readTsv("../labels.tsv");
const names = Object.keys(inquiryQuestions);
const matches = Object.fromEntries(names.map((name) => [name, 0]));
const times: number[] = [];
let changed = 0;
let cost = 0;

// 回答を正解ラベルと比べられる値に変換する（choiceは選んだ選択肢、noulは0.5を超えたら「はい」）
const answerOf = (a: any): string => (a.type === "choice" ? a.choice : a.noul > 0.5 ? "はい" : "いいえ");

for (const [id, receivedAt] of (await readTsv("../inquiries/received-at.tsv")).slice(1)) {
  const inquiry = await Bun.file(new URL(`../inquiries/${id}.txt`, import.meta.url)).text();
  const results = new Set<string>();
  for (let i = 0; i < repeats; i++) {
    const started = performance.now();
    const response = await openrouter.alpha.decisions.create({
      decisionsRequest: { model, state: { received_at: receivedAt, inquiry }, questions: inquiryQuestions as any },
    });
    times.push(performance.now() - started);
    cost += response.usage.cost ?? 0;
    const values = names.map((name) => answerOf(response.answers[name]));
    results.add(values.join());
    names.forEach((name, j) => {
      if (labels.some((l) => l[0] === id && l[header!.indexOf(name)] === values[j])) matches[name]!++;
    });
  }
  if (results.size > 1) changed++;
}

times.sort((a, b) => a - b);
console.log(`モデル: ${model}（各${repeats}回）`);
for (const name of names) console.log(`${name}: 正解との一致 ${(matches[name]! / repeats).toFixed(1)}/30`);
console.log(`回答が一度でも変わった問い合わせ: ${changed}/30`);
console.log(`所要時間の中央値: ${Math.round(times[times.length >> 1]!)}ms、利用料金の合計: ${cost.toFixed(6)} USD`);
