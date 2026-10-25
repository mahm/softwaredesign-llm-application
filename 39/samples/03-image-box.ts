// TypeSafe公式サイトの画像から、Create Accountリンクの有無と位置を調べる
import { readFile } from "node:fs/promises";
import { box, image, noul, TypeSafeClient } from "@typesafe-ai/sdk";

const screenshot = image(await readFile(new URL("../assets/typesafe-homepage.png", import.meta.url)));
const client = new TypeSafeClient();
const response = await client.systemOne({
  model: "jev-1.14.0",
  state: { screenshot },
  questions: {
    create_account_visible: noul("`screenshot`に「Create Account」というリンクはありますか"),
    create_account_link: box("`screenshot`にある「Create Account」というリンク"),
  },
}, {
  headers: { "TypeSafe-Danger-Zone-Images": "true" },
});
console.log(JSON.stringify(response, null, 2));
