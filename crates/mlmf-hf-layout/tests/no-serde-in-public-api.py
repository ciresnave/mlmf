"""Extract PUBLIC SIGNATURES ONLY -- pub fn/struct/enum/type up to `{` or `;`
-- and report any that name serde_json. A body may use it freely; a
signature may not."""
import io, glob, re, sys
bad = []
sigs = 0
for f in glob.glob("crates/mlmf-hf-layout/src/*.rs"):
    src = io.open(f, encoding="utf-8").read()
    for m in re.finditer(r"^\s*pub (?:fn|struct|enum|type|const)\b", src, re.M):
        start = m.start()
        end = len(src)
        for i in range(start, len(src)):
            if src[i] in "{;":
                end = i
                break
        sig = " ".join(src[start:end].split())
        sigs += 1
        if "serde_json" in sig:
            bad.append((f, sig[:100]))
    # public FIELDS too -- what mlmf-safetensors had to solve with pub(crate)
    for m in re.finditer(r"^\s*pub \w+\s*:[^,\n]*", src, re.M):
        sigs += 1
        if "serde_json" in m.group(0):
            bad.append((f, m.group(0).strip()[:100]))
print(f"  public signatures + fields scanned: {sigs}")
for f, s in bad:
    print(f"  !! serde_json in a public item: {f}: {s}")
sys.exit(1 if bad else 0)
