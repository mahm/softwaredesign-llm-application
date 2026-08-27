export function priceWithTax(
  unitPrice: number,
  quantity: number,
  taxRate = 0.1,
): number {
  return Math.floor(unitPrice * (1 + taxRate)) * quantity;
}
