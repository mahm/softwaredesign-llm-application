// 問い合わせ内容から参照すべき文書カテゴリ（collection）をnoulで判定し、検索対象を絞り込む
import { noul, TypeSafeClient } from "@typesafe-ai/sdk";
import { type Chunk, hybridSearch, loadData } from "./search.ts";

const { inquiry, chunks } = await loadData();

// 1. 絞り込みなしで全カテゴリのチャンクをハイブリッド検索する
const unfilteredHits = await hybridSearch(inquiry, chunks, 5);
console.log("--- 絞り込みなしのハイブリッド検索上位5件 ---");
for (const hit of unfilteredHits) {
  console.log(
    `[${hit.chunk.id}] (${hit.chunk.collection}) RRF=${hit.rrfScore.toFixed(4)} (Vec:${hit.vectorRank}位, BM25:${hit.bm25Rank}位) ${hit.chunk.title}`,
  );
}

// 2. 各文書カテゴリを検索対象に含めるべきかを1回のリクエストでJevに尋ねる
const client = new TypeSafeClient();
const routingResponse = await client.systemOne({
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

console.log("\n--- Jevによるカテゴリ判定結果（noul） ---");
console.log(JSON.stringify(routingResponse.answers, null, 2));

// noulの値が0.5以上のカテゴリに検索対象を絞り込む
const selectedCollections = (
  Object.keys(routingResponse.answers) as Array<Chunk["collection"]>
).filter((key) => routingResponse.answers[key].noul >= 0.5);

const filteredChunks = chunks.filter((c) => selectedCollections.includes(c.collection));
const filteredHits = await hybridSearch(inquiry, filteredChunks, 5);

console.log(`\n--- カテゴリ絞り込み後（${selectedCollections.join(", ")}）の上位5件 ---`);
for (const hit of filteredHits) {
  console.log(
    `[${hit.chunk.id}] (${hit.chunk.collection}) RRF=${hit.rrfScore.toFixed(4)} (Vec:${hit.vectorRank}位, BM25:${hit.bm25Rank}位) ${hit.chunk.title}`,
  );
}
