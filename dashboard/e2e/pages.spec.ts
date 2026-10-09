import { expect, test } from "@playwright/test";

const PAGES = [
  ["overview", "Serving LLM agents efficiently"],
  ["engines", "Engines (E1)"],
  ["caching", "Caching (E2, E6)"],
  ["scheduling", "llm-d scheduling (E3)"],
  ["hybrid", "Hybrid serving and cost (E4)"],
  ["quality", "Quality and cost (E5)"],
  ["context", "Context policies for agents (v0.4)"],
  ["gateway", "Context management in the gateway (v0.5)"],
  ["provenance", "Run provenance"],
] as const;

for (const scheme of ["light", "dark"] as const) {
  test.describe(`${scheme} mode`, () => {
    test.use({ colorScheme: scheme });
    for (const [id, heading] of PAGES) {
      test(`${id} renders from saved results`, async ({ page }) => {
        const errors: string[] = [];
        page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto(`/#/${id}`);
        await expect(page.getByRole("heading", { level: 2, name: heading })).toBeVisible();
        await expect(page.getByRole("link", { name: /Overview/ })).toBeVisible();
        if (id !== "overview" && id !== "provenance") {
          await expect(page.locator("svg.recharts-surface").first()).toBeVisible();
        }
        expect(errors).toEqual([]);
      });
    }
  });
}

test("table view toggles and shows CIs", async ({ page }) => {
  await page.goto("/#/hybrid");
  await page.getByRole("button", { name: "Table" }).first().click();
  await expect(page.getByRole("columnheader", { name: "95% CI" }).first()).toBeVisible();
});

test("provenance lists every exported run", async ({ page }) => {
  const index = await (await page.request.get("/data/index.json")).json();
  await page.goto("/#/provenance");
  for (const e of index.experiments) {
    await expect(page.getByRole("cell", { name: e.run_id })).toBeVisible();
  }
});

test("theme switch overrides the system setting", async ({ page }) => {
  await page.goto("/#/overview");
  await page.getByRole("radio", { name: "dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("radio", { name: "system" }).click();
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
});
