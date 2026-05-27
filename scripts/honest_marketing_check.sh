#!/usr/bin/env bash
# Mechanical honest-marketing firewall (ship-and-yank prevention).
# ERE throughout. Real runner: this script's own exit code gates CI.
# Finalized at S7 (adds the synthetic-without-disclaimer negative test).
set -uo pipefail

SRC="src/vlatrust"
fail=0

note() { printf '  %s\n' "$1"; }

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
#    disclaimer. (Negative-test fixture wired in at S7.)
if grep -rlE '"mode"[[:space:]]*:[[:space:]]*"synthetic"' bench_results report_samples 2>/dev/null \
     | while read -r f; do grep -qE 'disclaimer' "$f" || echo "$f"; done | grep -q .; then
  note "FAIL: a synthetic-mode artifact is missing a disclaimer."
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "honest-marketing: BLOCK"
  exit 1
fi
echo "honest-marketing: OK"
