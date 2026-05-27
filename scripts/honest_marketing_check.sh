#!/usr/bin/env bash
# Mechanical honest-marketing firewall (ship-and-yank prevention).
# ERE throughout. Real runner: this script's own exit code gates CI.
# `--selftest` proves the synthetic-without-disclaimer detector actually trips.
set -uo pipefail

SRC="src/vlatrust"
fail=0

note() { printf '  %s\n' "$1"; }

# Returns 0 (caught) iff a mode:synthetic artifact in $1 lacks a disclaimer.
synthetic_missing_disclaimer() {
  grep -rlE '"mode"[[:space:]]*:[[:space:]]*"synthetic"' "$1" 2>/dev/null \
    | while read -r f; do grep -qE 'disclaimer' "$f" || echo "$f"; done | grep -q .
}

# Negative test: the detector MUST catch a planted bad fixture and MUST clear a
# disclaimer'd one. Exits 0 only if both hold (so CI can assert the firewall works).
if [ "${1:-}" = "--selftest" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  printf '{"mode": "synthetic", "x": 1}\n' > "$tmp/bad.json"
  printf '{"mode": "synthetic", "disclaimer": "demo only"}\n' > "$tmp/good.json"
  if ! synthetic_missing_disclaimer "$tmp"; then
    echo "selftest FAIL: detector did not catch synthetic-without-disclaimer"; exit 1
  fi
  rm -f "$tmp/bad.json"
  if synthetic_missing_disclaimer "$tmp"; then
    echo "selftest FAIL: detector false-positived on a disclaimer'd fixture"; exit 1
  fi
  echo "honest-marketing selftest: OK"; exit 0
fi

# 1. No stdlib `random.` in production paths. A seeded numpy Generator
#    (np.random.default_rng / np.random.Generator) IS allowed for bootstrap;
#    the regex below excludes any `random.` preceded by `.` or an identifier
#    char, so `np.random.` never matches.
if grep -rnE '(^|[^.[:alnum:]_])random\.' "$SRC" --include='*.py'; then
  note "FAIL: stdlib random.* in a production path (use a seeded np.random.Generator)."
  fail=1
fi
if grep -rnE '^[[:space:]]*import[[:space:]]+random([[:space:]]|$)' "$SRC" --include='*.py'; then
  note "FAIL: bare 'import random' in a production path."
  fail=1
fi

# 2. Overclaim tokens in human-facing docs (README + docs/).
OVERCLAIM='fully automatic|real-world safe|guaranteed safe|永続|完全自動'
if grep -rnE "$OVERCLAIM" README.md docs 2>/dev/null; then
  note "FAIL: overclaim token in human-facing docs."
  fail=1
fi

# 3. mode:"synthetic" emitted without an accompanying disclaimer field.
#    A report JSON or fixture that declares synthetic data must also carry a
#    disclaimer. (The detector itself is exercised by `--selftest`.)
for scan in bench_results report_samples; do
  if [ -d "$scan" ] && synthetic_missing_disclaimer "$scan"; then
    note "FAIL: a synthetic-mode artifact in $scan/ is missing a disclaimer."
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "honest-marketing: BLOCK"
  exit 1
fi
echo "honest-marketing: OK"
