#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p vendor
restore() {
    local name="$1" url="$2" pin_file="$3" revision
    revision="$(cat "$pin_file")"
    if [[ -d "vendor/$name/.git" ]]; then
        [[ "$(git -C "vendor/$name" rev-parse HEAD)" == "$revision" ]] || {
            echo "Existing $name has a different revision; leaving it unchanged." >&2
            return 1
        }
        git -C "vendor/$name" diff --exit-code --quiet
        return
    fi
    if [[ -e "vendor/$name" ]]; then
        echo "Existing exported $name is preserved. Audit it; this command does not overwrite it." >&2
        return 1
    fi
    git clone --filter=blob:none "$url" "vendor/$name"
    git -C "vendor/$name" checkout --detach "$revision"
}
restore savanna https://github.com/Zymrael/savanna.git SAVANNA_COMMIT
restore synthetic_genomes_benchmark https://github.com/Georgakopoulos-Soares-lab/Synthetic_genomes_benchmark.git CNN_BENCHMARK_COMMIT
