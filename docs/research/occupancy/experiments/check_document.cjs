// Validate research math, local references, and statement-before-proof ordering.
const fs = require("fs");
const path = require("path");
const katex = require("katex");

function checkTheorems(raw) {
    const issues = [];
    const headings = [
        ...raw.matchAll(/^### \d+\.\d+ (?:定理|命题) \d+[^\n]*$/gm),
    ];
    if (!headings.length)
        issues.push("research document has no theorem statements");
    for (const heading of headings) {
        const rest = raw.slice(heading.index + heading[0].length);
        const boundary = rest.search(/^#{1,3} /m);
        const block = boundary < 0 ? rest : rest.slice(0, boundary);
        const statement = block.search(/\*\*(?:定理|命题)陈述。\*\*/);
        const proof = block.indexOf("**证明。**");
        if (statement < 0 || proof < 0 || statement >= proof) {
            issues.push(`${heading[0]}: statement must precede proof`);
        }
    }
    return { count: headings.length, issues };
}

function checkDocument(raw, root) {
    const issues = [];
    const unavailableExternalLinks = [];
    let mathCount = 0;
    let links = 0;
    const prose = raw.replace(/```[\s\S]*?```/g, "");
    for (const match of prose.matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)) {
        const target = match[1];
        if (/^(https?:|#)/.test(target)) continue;
        links++;
        const resolved = path.resolve(root, target.split("#")[0]);
        if (!fs.existsSync(resolved)) {
            const external = ["Awesome-OPD-Paper", "MirrorOPD"].some((repo) =>
                resolved.startsWith(
                    path.resolve(root, "../../../..", repo) + path.sep,
                ),
            );
            if (external) unavailableExternalLinks.push(target);
            else issues.push(`missing link: ${target}`);
        }
    }
    const definitions = new Set(
        [...raw.matchAll(/^\[\^([a-z-]+)\]:/gm)].map((m) => m[1]),
    );
    for (const match of raw.matchAll(/\[\^([^\]]+)\](?!:)/g)) {
        if (!definitions.has(match[1]))
            issues.push(`missing footnote: ${match[1]}`);
    }
    const displays = [...prose.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(
        (m) => m[1],
    );
    const inlines = [
        ...prose
            .replace(/\$\$[\s\S]*?\$\$/g, "")
            .matchAll(/(?<!\\)\$([^$\n]+?)\$/g),
    ].map((m) => m[1]);
    for (const [expressions, displayMode] of [
        [displays, true],
        [inlines, false],
    ]) {
        for (const math of expressions) {
            mathCount++;
            try {
                katex.renderToString(math, {
                    throwOnError: true,
                    strict: "ignore",
                    displayMode,
                });
            } catch (error) {
                issues.push(`math ${mathCount}: ${error.message}`);
            }
        }
    }
    for (const match of raw.matchAll(/```pseudocode\n([\s\S]*?)```/g)) {
        const pseudo = match[1];
        if (!pseudo.includes("\\\\{\\small\\textit{Semantics:")) {
            issues.push("caption semantics line break missing");
        }
        for (const [open, close] of [
            ["FOR", "ENDFOR"],
            ["IF", "ENDIF"],
        ]) {
            const count = (token) =>
                (
                    pseudo.match(
                        new RegExp("\\\\" + token + "(?:\\{|\\n)", "g"),
                    ) || []
                ).length;
            if (count(open) !== count(close))
                issues.push(`${open} block mismatch`);
        }
    }
    const theorems = checkTheorems(raw);
    issues.push(...theorems.issues);
    return {
        math_expressions: mathCount,
        theorem_statements: theorems.count,
        local_links: links,
        footnotes: definitions.size,
        unavailable_external_links: unavailableExternalLinks,
        issues,
    };
}

if (require.main === module) {
    const root = path.resolve(__dirname, "..");
    const file = path.join(root, "2026-10-17-OPD访问测度与收敛机制研究.md");
    const result = checkDocument(fs.readFileSync(file, "utf8"), root);
    fs.writeFileSync(
        path.join(root, "results/document-check.json"),
        JSON.stringify(result, null, 2) + "\n",
    );
    console.log(result);
    if (result.issues.length) process.exitCode = 1;
}
module.exports = { checkDocument, checkTheorems };
