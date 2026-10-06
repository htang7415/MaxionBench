import { useEffect, useState } from "react";
import { useData } from "./data";
import { Caching, Engines, Hybrid, Overview, Provenance, Quality, Scheduling } from "./pages/pages";

const PAGES = [
  { id: "overview", title: "Overview", Page: Overview },
  { id: "engines", title: "Engines", Page: Engines },
  { id: "caching", title: "Caching", Page: Caching },
  { id: "scheduling", title: "llm-d scheduling", Page: Scheduling },
  { id: "hybrid", title: "Hybrid serving", Page: Hybrid },
  { id: "quality", title: "Quality and cost", Page: Quality },
  { id: "provenance", title: "Provenance", Page: Provenance },
] as const;

function useHashPage(): string {
  const read = () => window.location.hash.replace("#/", "") || "overview";
  const [page, setPage] = useState(read);
  useEffect(() => {
    const onHash = () => setPage(read());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return page;
}

export default function App() {
  const { data, error } = useData();
  const page = useHashPage();
  const current = PAGES.find((p) => p.id === page) ?? PAGES[0];
  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 md:flex-row md:px-8">
      <nav className="md:w-48 md:shrink-0" aria-label="Pages">
        <div className="mb-4 text-[15px] font-semibold">MaxionBench</div>
        <ul className="flex flex-wrap gap-1 md:flex-col">
          {PAGES.map((p) => (
            <li key={p.id}>
              <a
                href={`#/${p.id}`}
                aria-current={p.id === current.id ? "page" : undefined}
                className="block rounded-md px-2.5 py-1.5 text-sm"
                style={p.id === current.id
                  ? { background: "var(--surface)", color: "var(--ink)", fontWeight: 600, boxShadow: "0 0 0 1px var(--border)" }
                  : { color: "var(--ink-2)" }}
              >
                {p.title}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <main className="min-w-0 flex-1">
        {error && <p role="alert">Could not load results: {error}. Run <code>npm run data</code>.</p>}
        {!error && !data && <p style={{ color: "var(--muted)" }}>Loading results…</p>}
        {data && <current.Page data={data} />}
      </main>
    </div>
  );
}
