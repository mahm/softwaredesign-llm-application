import { CopilotKit } from "@copilotkit/react-core";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "arXiv Slide Generator",
  description: "arXiv論文からスライドを対話的に生成するエージェント",
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
