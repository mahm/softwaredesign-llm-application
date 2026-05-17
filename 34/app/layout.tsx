import { CopilotKit } from "@copilotkit/react-core";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "ACP Multi-Agent Demo",
  description: "ACPXでClaude CodeとCodexを呼び出すDeep Agentsデモ",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit">{children}</CopilotKit>
      </body>
    </html>
  );
}
