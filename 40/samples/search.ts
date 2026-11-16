// OpenRouterのEmbeddings APIによるベクトル検索とBM25全文検索をRRFで統合するハイブリッド検索ヘルパー
import { readFile } from "node:fs/promises";
import { OpenRouter } from "@openrouter/sdk";

export interface Chunk {
  id: string;
  collection: "it_manual" | "security_policy" | "hr_general" | "incident_log";
  section_id: string;
  chunk_index: number;
  title: string;
  text: string;
}

export interface SearchHit {
  chunk: Chunk;
  rrfScore: number;
  vectorRank: number;
  bm25Rank: number;
}

const RRF_K = 60;
const segmenter = new Intl.Segmenter("ja", { granularity: "word" });

export function getOpenRouterClient(): OpenRouter {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
  return new OpenRouter({ apiKey });
}

export async function loadData(): Promise<{ inquiry: string; chunks: Chunk[] }> {
  const inquiry = await readFile(new URL("../inquiries/sdk-example.txt", import.meta.url), "utf8");
  const rawChunks = await readFile(new URL("../docs/chunks.json", import.meta.url), "utf8");
  const chunks = JSON.parse(rawChunks) as Chunk[];
  return { inquiry: inquiry.trim(), chunks };
}

function tokenize(text: string): string[] {
  const tokens: string[] = [];
  for (const seg of segmenter.segment(text.toLowerCase())) {
    if (seg.isWordLike) tokens.push(seg.segment);
  }
  return tokens;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function computeBm25Scores(query: string, docs: string[]): number[] {
  const k1 = 1.2;
  const b = 0.75;
  const queryTokens = Array.from(new Set(tokenize(query)));
  const docTokensList = docs.map(tokenize);
  const avgDl = docTokensList.reduce((sum, t) => sum + t.length, 0) / docs.length;

  const df = new Map<string, number>();
  for (const term of queryTokens) {
    let count = 0;
    for (const tokens of docTokensList) {
      if (tokens.includes(term)) count++;
    }
    df.set(term, count);
  }

  return docTokensList.map((tokens) => {
    let score = 0;
    for (const term of queryTokens) {
      const tf = tokens.filter((t) => t === term).length;
      if (tf === 0) continue;
      const n = df.get(term) ?? 0;
      const idf = Math.log(1 + (docs.length - n + 0.5) / (n + 0.5));
      score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (tokens.length / avgDl))));
    }
    return score;
  });
}

export async function hybridSearch(
  query: string,
  chunks: Chunk[],
  topK = 5,
): Promise<SearchHit[]> {
  const openRouter = getOpenRouterClient();
  const docs = chunks.map((c) => `${c.title}\n${c.text}`);

  // 1. OpenRouter Embeddings APIによるベクトル検索
  const embResponse = await openRouter.embeddings.generate({
    requestBody: {
      model: "openai/text-embedding-3-small",
      input: [query, ...docs],
      encodingFormat: "float",
    },
  });
  if (typeof embResponse === "string") {
    throw new Error("Unexpected string response from embeddings API");
  }
  const vectors = embResponse.data.map((d) => {
    if (!Array.isArray(d.embedding)) throw new Error("Expected float array embedding");
    return d.embedding;
  });
  const queryVector = vectors[0];
  const vectorScores = vectors.slice(1).map((v) => cosineSimilarity(queryVector, v));

  // 2. Okapi BM25による全文検索
  const bm25Scores = computeBm25Scores(query, docs);

  // 3. それぞれの順位（1始まり）を算出し、RRF（k=60）で統合する
  const vectorOrder = chunks
    .map((_, idx) => idx)
    .sort((a, b) => vectorScores[b] - vectorScores[a]);
  const bm25Order = chunks
    .map((_, idx) => idx)
    .sort((a, b) => bm25Scores[b] - bm25Scores[a]);

  const hits: SearchHit[] = chunks.map((chunk, idx) => {
    const vectorRank = vectorOrder.indexOf(idx) + 1;
    const bm25Rank = bm25Order.indexOf(idx) + 1;
    const rrfScore = 1 / (RRF_K + vectorRank) + 1 / (RRF_K + bm25Rank);
    return { chunk, rrfScore, vectorRank, bm25Rank };
  });

  return hits.sort((a, b) => b.rrfScore - a.rrfScore).slice(0, topK);
}
