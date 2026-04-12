// server.ts
import { DeepAgentsServer, ACPFilesystemBackend } from "deepagents-acp";

const server = new DeepAgentsServer({
  agents: {
    name: "coding-assistant",
    description: "Deep Agentsベースのコーディングアシスタント",
    model: "claude-sonnet-4-6",
  },
  debug: true,
});

// ACPFilesystemBackendのvirtualMode未対応を回避
// virtualModeを有効にし、resolveAbsPathが仮想パスを実パスに変換するよう修正
const originalCreateBackend = server.createBackend.bind(server);
server.createBackend = function (config) {
  const backend = originalCreateBackend(config);
  if (backend instanceof ACPFilesystemBackend) {
    backend.virtualMode = true;
    backend.resolveAbsPath = function (filePath) {
      return this.resolvePath(filePath);
    };
  }
  return backend;
};

await server.start();
