// Validate research math, local references, and statement-before-proof ordering.
const fs = require("fs");
const path = require("path");
const katex = require("katex");

function withoutCode(raw) {
    return raw.replace(/```[\s\S]*?```/g, "");
}

function anchorsFor(raw) {
    const anchors = new Set();
    for (const match of withoutCode(raw).matchAll(/^#{1,6} ([^\n]+)$/gm)) {
        const base = match[1]
            .toLowerCase()
            .replace(/[^\p{L}\p{N}\p{M} _-]/gu, "")
            .replace(/ /g, "-");
        let anchor = base;
        let suffix = 1;
        while (anchors.has(anchor)) anchor = `${base}-${suffix++}`;
        anchors.add(anchor);
    }
    return anchors;
}

function localTargets(raw) {
    return [...withoutCode(raw).matchAll(/!?\[[^\]]*\]\(([^)]+)\)/g)]
        .map((m) => m[1])
        .filter((target) => !/^[a-z][\w+.-]*:/i.test(target));
}

function checkTheorems(raw, { requireTheorems = true } = {}) {
    const issues = [];
    const headings = [
        ...raw.matchAll(/^### \d+\.\d+ (?:定理|命题) \d+[^\n]*$/gm),
    ];
    if (!headings.length && requireTheorems)
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
    const numbers = headings.map((h) =>
        Number(h[0].match(/(?:定理|命题) (\d+)/)[1]),
    );
    if (numbers.some((number, index) => number !== index + 1)) {
        issues.push("theorem numbers must be consecutive in reading order");
    }
    const known = new Set(numbers);
    // Linked references are checked at their destination, which may use its own numbering.
    const unlinked = withoutCode(raw).replace(/\[[^\]]*\]\([^)]+\)/g, "");
    for (const reference of unlinked.matchAll(/(?:定理|命题) (\d+)/g)) {
        if (!known.has(Number(reference[1]))) {
            issues.push(`unknown theorem reference: ${reference[0]}`);
        }
    }
    return { count: headings.length, issues };
}

function checkDocument(raw, root, options = {}) {
    const issues = [];
    const unavailableExternalLinks = [];
    let mathCount = 0;
    let links = 0;
    const prose = withoutCode(raw);
    const mermaids = [...raw.matchAll(/```mermaid\n([\s\S]*?)```/g)].map(
        (m) => m[1],
    );
    const mermaidMath = mermaids.flatMap((diagram) =>
        [...diagram.matchAll(/\$\$([\s\S]*?)\$\$/g)].map((m) => m[1]),
    );
    const anchors = anchorsFor(raw);
    for (const target of localTargets(raw)) {
        if (target.startsWith("#")) {
            if (!anchors.has(decodeURIComponent(target.slice(1))))
                issues.push(`missing anchor: ${target}`);
            continue;
        }
        links++;
        const [filePart, anchor] = target.split("#");
        const resolved = path.resolve(root, decodeURIComponent(filePart));
        if (!fs.existsSync(resolved)) {
            const external = [
                "Awesome-OPD-Paper",
                "Awesome-Paper",
                "MirrorOPD",
            ].some((repo) =>
                resolved.startsWith(
                    path.resolve(root, "../../../..", repo) + path.sep,
                ),
            );
            if (external) unavailableExternalLinks.push(target);
            else issues.push(`missing link: ${target}`);
        } else if (anchor && path.extname(resolved) === ".md") {
            const destination = anchorsFor(fs.readFileSync(resolved, "utf8"));
            if (!destination.has(decodeURIComponent(anchor))) {
                issues.push(`missing linked anchor: ${target}`);
            }
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
        [mermaidMath, true],
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
    const theorems = checkTheorems(raw, options);
    issues.push(...theorems.issues);
    return {
        math_expressions: mathCount,
        mermaid_diagrams: mermaids.length,
        mermaid_math_expressions: mermaidMath.length,
        theorem_statements: theorems.count,
        local_links: links,
        footnotes: definitions.size,
        unavailable_external_links: unavailableExternalLinks,
        issues,
    };
}

function checkCollection(
    root,
    entry = "2026-10-17-OPD访问测度与收敛机制研究.md",
) {
    const issues = [];
    const files = fs
        .readdirSync(root)
        .filter((name) => name.endsWith(".md"))
        .sort();
    if (!files.length) issues.push("no research documents selected");
    if (!files.includes(entry)) issues.push(`missing entry document: ${entry}`);
    const source = new Map(
        files.map((name) => [
            name,
            fs.readFileSync(path.join(root, name), "utf8"),
        ]),
    );
    const documents = files.map((name) => {
        const result = checkDocument(source.get(name), root, {
            requireTheorems: false,
        });
        issues.push(...result.issues.map((issue) => `${name}: ${issue}`));
        return {
            document: name,
            lines: source.get(name).trimEnd().split("\n").length,
            ...result,
        };
    });
    const reachable = new Set();
    const pending = files.includes(entry) ? [entry] : [];
    while (pending.length) {
        const name = pending.pop();
        if (reachable.has(name)) continue;
        reachable.add(name);
        for (const target of localTargets(source.get(name))) {
            const filePart = decodeURIComponent(target.split("#")[0]);
            if (!filePart) continue;
            const relative = path.relative(root, path.resolve(root, filePart));
            if (source.has(relative) && !reachable.has(relative))
                pending.push(relative);
        }
    }
    for (const name of files) {
        if (!reachable.has(name))
            issues.push(`document is unreachable from entry: ${name}`);
    }
    const sum = (key) => documents.reduce((total, doc) => total + doc[key], 0);
    if (!sum("theorem_statements"))
        issues.push("research collection has no theorem statements");
    const footnotes = new Set(
        [...source.values()].flatMap((raw) =>
            [...raw.matchAll(/^\[\^([a-z-]+)\]:/gm)].map((m) => m[1]),
        ),
    );
    return {
        entry,
        document_count: documents.length,
        math_expressions: sum("math_expressions"),
        mermaid_diagrams: sum("mermaid_diagrams"),
        mermaid_math_expressions: sum("mermaid_math_expressions"),
        theorem_statements: sum("theorem_statements"),
        local_links: sum("local_links"),
        footnotes: footnotes.size,
        unavailable_external_links: [
            ...new Set(
                documents.flatMap((doc) => doc.unavailable_external_links),
            ),
        ],
        documents,
        issues,
    };
}

if (require.main === module) {
    const root = path.resolve(__dirname, "..");
    const result = checkCollection(root);
    fs.writeFileSync(
        path.join(root, "results/document-check.json"),
        JSON.stringify(result, null, 2) + "\n",
    );
    const { documents, ...summary } = result;
    console.log(summary);
    if (result.issues.length) process.exitCode = 1;
}
module.exports = { checkCollection, checkDocument, checkTheorems };
