import {
  readTraceSnapshot,
  readTraceSnapshots,
} from "../../../agent/trace-store";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const agent = url.searchParams.get("agent");
  const session = url.searchParams.get("session");
  const turnParam = url.searchParams.get("turn");
  const turn = turnParam ? Number(turnParam) : undefined;

  if (!agent || !session) {
    return Response.json(
      { error: "agent and session are required" },
      { status: 400 },
    );
  }

  if (turnParam && (!Number.isInteger(turn) || Number(turn) <= 0)) {
    return Response.json(
      { error: "turn must be a positive integer" },
      { status: 400 },
    );
  }

  if (turn !== undefined) {
    const snapshot = await readTraceSnapshot(agent, session, turn);
    return Response.json({
      found: Boolean(snapshot),
      snapshot,
      snapshots: snapshot ? [snapshot] : [],
    });
  }

  const snapshots = await readTraceSnapshots(agent, session);
  return Response.json({ found: snapshots.length > 0, snapshots });
}
