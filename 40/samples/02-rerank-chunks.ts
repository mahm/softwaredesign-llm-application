// ハイブリッド検索の候補チャンクを1回のリクエストで評価し、期待値スコアでリランキングする
import { score, type ScoreQuestion, TypeSafeClient } from "@typesafe-ai/sdk";
import { hybridSearch, loadData } from "./search.ts";

const { inquiry, chunks } = await loadData();

// ステップ1で絞り込まれた2カテゴリ（it_manual, security_policy）から候補を取得する
const targetChunks = chunks.filter(
  (c) => c.collection === "it_manual" || c.collection === "security_policy",
);
const initialHits = await hybridSearch(inquiry, targetChunks, 5);

// 候補チャンクを構造化してstateに渡し、各チャンクの有用性をscoreで同時に尋ねる
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

const client = new TypeSafeClient();
const response = await client.systemOne({
  model: "jev-1.14.0",
  state: { inquiry, candidates },
  questions,
});

// Jevが返すscore（0〜2の期待値）を5倍し、0〜10点満点（無関係:0点 / 一部関連:5点 / 直接の回答:10点）に換算して並べ替える
const reranked = initialHits
  .map((hit, idx) => {
    const ans = response.answers[hit.chunk.id];
    const pointScore = ans.score * 5;
    return {
      initialRank: idx + 1,
      hit,
      pointScore,
      probabilities: ans.probabilities,
    };
  })
  .sort((a, b) => b.pointScore - a.pointScore);

console.log("--- Jevによるリランキング結果（10点満点換算） ---");
for (const item of reranked) {
  const p = item.probabilities;
  console.log(
    `[${item.hit.chunk.id}] スコア=${item.pointScore.toFixed(2)}点 (初期順位:${item.initialRank}位) ` +
      `[直接の回答:${p["2"].toFixed(2)}, 一部関連:${p["1"].toFixed(2)}, 無関係:${p["0"].toFixed(2)}] ` +
      `${item.hit.chunk.title}`,
  );
}
