import { useEffect, useState } from "react";
import { useData, type Data } from "./data";
import { Caching, ContextPolicies, Engines, GatewayContext, Hybrid, Overview, Provenance, Quality, Scheduling } from "./pages/pages";

const REPO = "https://github.com/htang7415/MaxionBench";

interface PageDef {
  id: string;
  title: string;
  Page: (props: { data: Data }) => React.JSX.Element;
}

// Grouped by topic.
const GROUPS: { label: string | null; pages: PageDef[] }[] = [
  { label: null, pages: [{ id: "overview", title: "Overview", Page: Overview }] },
  { label: "Gateway", pages: [{ id: "gateway", title: "Gateway context", Page: GatewayContext }] },
  { label: "Agents", pages: [{ id: "context", title: "Agent context", Page: ContextPolicies }] },
  {
    label: "Serving",
    pages: [
      { id: "engines", title: "Engines", Page: Engines },
      { id: "caching", title: "Caching", Page: Caching },
      { id: "scheduling", title: "llm-d scheduling", Page: Scheduling },
      { id: "hybrid", title: "Hybrid serving", Page: Hybrid },
      { id: "quality", title: "Quality and cost", Page: Quality },
    ],
  },
  { label: "Data", pages: [{ id: "provenance", title: "Provenance", Page: Provenance }] },
];
const PAGES = GROUPS.flatMap((g) => g.pages);

function useHashPage(): string {
  const read = () => window.location.hash.replace("#/", "") || "overview";
  const [page, setPage] = useState(read);
  useEffect(() => {
    const onHash = () => {
      setPage(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return page;
}

type ThemeChoice = "system" | "light" | "dark";

function readChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem("theme");
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

/** Light / dark / system; the CSS tokens follow `data-theme`, or the OS when it is unset. */
function ThemeSwitch() {
  const [choice, setChoice] = useState<ThemeChoice>(readChoice);
  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", choice);
    try {
      localStorage.setItem("theme", choice);
    } catch {
      /* storage unavailable: the choice lasts for this visit */
    }
  }, [choice]);
  return (
    <div role="radiogroup" aria-label="Color theme" className="flex rounded-full border p-0.5 text-xs"
      style={{ borderColor: "var(--border)" }}>
      {(["light", "system", "dark"] as const).map((c) => (
        <button key={c} type="button" role="radio" aria-checked={choice === c} onClick={() => setChoice(c)}
          className="rounded-full px-2.5 py-1 capitalize"
          style={choice === c ? { background: "var(--ink)", color: "var(--page)" } : { color: "var(--ink-2)" }}>
          {c}
        </button>
      ))}
    </div>
  );
}

export default function App() {
  const { data, error } = useData();
  const page = useHashPage();
  const current = PAGES.find((p) => p.id === page) ?? PAGES[0];
  const latest = data?.index.map((e) => e.finished_at).sort().at(-1);
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b backdrop-blur"
        style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--page) 85%, transparent)" }}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3 md:px-8">
          <a href="#/overview" className="text-[15px] font-semibold tracking-tight">MaxionBench</a>
          <div className="ml-auto flex items-center gap-3">
            <a href={REPO} className="hidden text-sm sm:inline" style={{ color: "var(--ink-2)" }}>GitHub</a>
            <ThemeSwitch />
          </div>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-8 px-4 py-8 md:flex-row md:px-8">
        <nav className="md:w-52 md:shrink-0" aria-label="Pages">
          <div className="flex flex-wrap gap-x-4 gap-y-3 md:sticky md:top-20 md:flex-col">
            {GROUPS.map((g) => (
              <div key={g.label ?? "home"}>
                {g.label && (
                  <div className="mb-1 px-2.5 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
                    {g.label}
                  </div>
                )}
                <ul className="flex flex-wrap gap-1 md:flex-col">
                  {g.pages.map((p) => (
                    <li key={p.id}>
                      <a href={`#/${p.id}`} aria-current={p.id === current.id ? "page" : undefined}
                        className="block rounded-md px-2.5 py-1.5 text-sm"
                        style={p.id === current.id
                          ? { background: "var(--surface)", color: "var(--ink)", fontWeight: 600, boxShadow: "0 0 0 1px var(--border)" }
                          : { color: "var(--ink-2)" }}>
                        {p.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </nav>
        <main className="min-w-0 flex-1">
          {error && <p role="alert">Could not load results: {error}. Run <code>npm run data</code>.</p>}
          {!error && !data && <p style={{ color: "var(--muted)" }}>Loading results…</p>}
          {data && <current.Page data={data} />}
        </main>
      </div>

      <footer className="border-t" style={{ borderColor: "var(--border)" }}>
        <div className="mx-auto flex max-w-7xl flex-wrap gap-x-6 gap-y-1 px-4 py-6 text-xs md:px-8" style={{ color: "var(--muted)" }}>
          <span>Every number comes from a saved result bundle; intervals are 95% CIs.</span>
          {latest && <span>Latest run {latest.slice(0, 10)}</span>}
          <a href={`${REPO}/blob/main/ARCHITECTURE.md`}>Architecture</a>
          <a href={`${REPO}/blob/main/LICENSE`}>License</a>
        </div>
      </footer>
    </div>
  );
}
