// スコア閾値でチャンクを選別し、同一セクションの前後チャンクを再構成してOpenRouterで回答を生成する
import { noul, score, type ScoreQuestion, TypeSafeClient } from "@typesafe-ai/sdk";
import { type Chunk, getOpenRouterClient, hybridSearch, loadData } from "./search.ts";

const SCORE_THRESHOLD = 5.0; // 10点満点換算で5.0点（「一部関連」相当）以上を採用
const { inquiry, chunks } = await loadData();
const client = new TypeSafeClient();
const openRouter = getOpenRouterClient();

// 1. Jev（noul）で検索対象のカテゴリを絞り込む
const routing = await client.systemOne({
  model: "jev-1.14.0",
  state: { inquiry },
  questions: {
    it_manual: noul(
      "`inquiry`に回答するために「IT機器・ネットワーク利用手順書（PCやVPNの初期設定・操作・トラブル対処）」を検索する必要がありますか",
    ),
    security_policy: noul(
      "`inquiry`に回答するために「情報セキュリティ規定（社外からの接続条件や利用ルール）」を検索する必要がありますか",
    ),
    hr_general: noul(
      "`inquiry`に回答するために「総務・労務ガイド（手当の支給条件や機器貸与の事前申請手続き）」を検索する必要がありますか",
    ),
    incident_log: noul(
      "`inquiry`に回答するために「過去の解消済み障害・定期メンテナンス完了履歴」を検索する必要がありますか",
    ),
  },
});

const selectedCollections = (
  Object.keys(routing.answers) as Array<Chunk["collection"]>
).filter((key) => routing.answers[key].noul >= 0.5);
const targetChunks = chunks.filter((c) => selectedCollections.includes(c.collection));

// 2. ハイブリッド検索の上位候補をJev（score）で一括リランキングする
const initialHits = await hybridSearch(inquiry, targetChunks, 5);
const candidates = initialHits.map((h) => ({
  id: h.chunk.id,
  title: h.chunk.title,
  text: h.chunk.text,
}));

const rubric = [
  "無関係（質問への回答になっていない、または対象者の前提条件が異なる）",
  "一部関連（質問の一部に答えている、または原因・前提条件は分かるが具体的な手順はない）",
  "直接の回答（質問に対する具体的な設定手順や解決策が書かれている）",
] as const;

const questions: Record<string, ScoreQuestion<typeof rubric>> = {};
for (let i = 0; i < candidates.length; i++) {
  questions[candidates[i].id] = score(
    `\`candidates[${i}]\`は、\`inquiry\`に回答するうえでどの程度役に立ちますか`,
    rubric,
  );
}

const rerankRes = await client.systemOne({
  model: "jev-1.14.0",
  state: { inquiry, candidates },
  questions,
});

const scoredChunks = initialHits
  .map((hit) => ({
    chunk: hit.chunk,
    pointScore: rerankRes.answers[hit.chunk.id].score * 5,
  }))
  .sort((a, b) => b.pointScore - a.pointScore);

// 3. スコア閾値で選別し、すべて閾値未満なら最上位1件のみを残す
const aboveThreshold = scoredChunks.filter((item) => item.pointScore >= SCORE_THRESHOLD);
const selected = aboveThreshold.length > 0 ? aboveThreshold : scoredChunks.slice(0, 1);

// 4. 採用されたチャンクと同一セクション（section_id）の前後チャンクを結合して再構成する
const reconstructedMap = new Map<string, Chunk>();
for (const { chunk } of selected) {
  const neighbors = chunks
    .filter(
      (c) =>
        c.section_id === chunk.section_id &&
        Math.abs(c.chunk_index - chunk.chunk_index) <= 1,
    )
    .sort((a, b) => a.chunk_index - b.chunk_index);
  for (const neighbor of neighbors) {
    reconstructedMap.set(neighbor.id, neighbor);
  }
}
const reconstructedChunks = Array.from(reconstructedMap.values());

console.log("--- 再構成されたチャンク一覧 ---");
for (const c of reconstructedChunks) {
  console.log(`[${c.id}] (section:${c.section_id} #${c.chunk_index}) ${c.title}`);
}

// 5. 再構成したコンテキストを使ってOpenRouter経由で回答を生成する
const contextText = reconstructedChunks
  .map((c) => `【${c.title}】\n${c.text}`)
  .join("\n\n");

const completion = await openRouter.chat.send({
  chatRequest: {
    model: "deepseek/deepseek-v4.1-flash-1017",
    messages: [
      {
        role: "system",
        content:
          "あなたは社内ヘルプデスクの担当者です。与えられた参考資料のみを根拠にして、問い合わせに対する回答を簡潔に作成してください。参考資料に書かれていない事項は推測で補わないでください。",
      },
      {
        role: "user",
        content: `## 参考資料\n${contextText}\n\n## 問い合わせ内容\n${inquiry}`,
      },
    ],
    maxCompletionTokens: 600,
    reasoningEffort: "none",
    temperature: 0,
    stream: false,
  },
});

if (Symbol.asyncIterator in completion) {
  throw new Error("Expected a non-streaming response");
}

console.log("\n--- 生成された回答 ---");
console.log(completion.choices[0]?.message.content);
