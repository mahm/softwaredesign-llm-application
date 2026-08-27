import { describe, expect, test } from "bun:test";

import { priceWithTax } from "./price";

describe("priceWithTax", () => {
  test("小計へ税率を適用してから小数部分を切り捨てる", () => {
    expect(priceWithTax(105, 3)).toBe(346);
  });

  test("異なる税率を指定できる", () => {
    expect(priceWithTax(199, 2, 0.08)).toBe(429);
  });
});
