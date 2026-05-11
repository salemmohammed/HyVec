#!/bin/bash
# =============================================================
#  benchmark.sh
#  Run this anytime you want to record and compare results.
#  Results are appended to results.csv automatically.
#
#  Usage:
#    ./benchmark.sh                     (full run)
#    ./benchmark.sh --search-only       (skip build, just search)
# =============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_CSV="../../results.csv"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
MACHINE=$(hostname)
THREADS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "1")
SEARCH_ONLY=false

for arg in "$@"; do
  [[ "$arg" == "--search-only" ]] && SEARCH_ONLY=true
done

echo "========================================================"
echo "  Benchmark — $(date '+%Y-%m-%d %H:%M')"
echo "  Machine: $MACHINE | Threads: $THREADS"
echo "========================================================"

# ── Detect OS and set compile flags ──────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  CXX_FLAGS="-O3 -std=c++17"
  THREAD_FLAGS="-lpthread"
else
  CXX_FLAGS="-O3 -std=c++17 -march=native -fopenmp"
  THREAD_FLAGS="-lpthread -fopenmp"
fi

# ── Step 1: Baseline ─────────────────────────────────────────
echo ""
echo ">>> [1/4] Baseline HNSW..."
cd ../baseline
g++ $CXX_FLAGS sift1m_search.cpp -o sift1m_search -I../../hnswlib 2>/dev/null
BASELINE_OUT=$(cd .. && ./baseline/sift1m_search 2>/dev/null)
BASELINE_RECALL=$(echo "$BASELINE_OUT" | grep "Recall@1" | awk '{print $2}')
BASELINE_QPS=$(echo "$BASELINE_OUT"    | grep "QPS"      | awk '{print $2}')
BASELINE_BUILD=$(echo "$BASELINE_OUT"  | grep "Build time" | awk '{print $3}' | tr -d 's')
echo "  Recall=$BASELINE_RECALL  QPS=$BASELINE_QPS  Build=${BASELINE_BUILD}s"
cd ../approach2_attribute

# ── Step 2: Cluster ───────────────────────────────────────────
if [ "$SEARCH_ONLY" = false ]; then
  echo ""
  echo ">>> [2/4] Clustering (K=1000)..."
  python3 generate_attributes.py 2>/dev/null | grep -E "cluster|time|Loaded"
fi

# ── Step 3: Build indexes ─────────────────────────────────────
if [ "$SEARCH_ONLY" = false ]; then
  echo ""
  echo ">>> [3/4] Building indexes..."
  g++ $CXX_FLAGS build_indexes.cpp -o build_indexes -I../../hnswlib 2>/dev/null
  BUILD_OUT=$(./build_indexes 2>/dev/null)
  CLUSTER_BUILD=$(echo "$BUILD_OUT" | grep "Total build time" | awk '{print $4}' | tr -d 's')
  echo "  Build time: ${CLUSTER_BUILD}s"
else
  CLUSTER_BUILD="skipped"
fi

# ── Step 4: Search sweep ──────────────────────────────────────
echo ""
echo ">>> [4/4] Search sweep..."
g++ $CXX_FLAGS search.cpp -o search -I../../hnswlib 2>/dev/null
SEARCH_OUT=$(./search 2>/dev/null)

TOP1_RECALL=$(echo  "$SEARCH_OUT" | awk '/^1 /{print $2}')
TOP1_QPS=$(echo     "$SEARCH_OUT" | awk '/^1 /{print $3}')
TOP3_RECALL=$(echo  "$SEARCH_OUT" | awk '/^3 /{print $2}')
TOP3_QPS=$(echo     "$SEARCH_OUT" | awk '/^3 /{print $3}')
TOP5_RECALL=$(echo  "$SEARCH_OUT" | awk '/^5 /{print $2}')
TOP5_QPS=$(echo     "$SEARCH_OUT" | awk '/^5 /{print $3}')
TOP10_RECALL=$(echo "$SEARCH_OUT" | awk '/^10 /{print $2}')
TOP10_QPS=$(echo    "$SEARCH_OUT" | awk '/^10 /{print $3}')
TOP20_RECALL=$(echo "$SEARCH_OUT" | awk '/^20 /{print $2}')
TOP20_QPS=$(echo    "$SEARCH_OUT" | awk '/^20 /{print $3}')
TOP50_RECALL=$(echo "$SEARCH_OUT" | awk '/^50 /{print $2}')
TOP50_QPS=$(echo    "$SEARCH_OUT" | awk '/^50 /{print $3}')

# ── Print comparison table ────────────────────────────────────
echo ""
echo "========================================================"
echo "  RESULTS"
echo "========================================================"
printf "%-25s %-12s %-12s %-12s\n" "Method" "Recall@1" "QPS" "Build(s)"
echo "--------------------------------------------------------"
printf "%-25s %-12s %-12s %-12s\n" "Baseline HNSW"     "$BASELINE_RECALL" "$BASELINE_QPS"  "$BASELINE_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "ANN-benchmarks*"   "0.9500"           "28021"          "N/A"
echo "--------------------------------------------------------"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=1"   "$TOP1_RECALL"     "$TOP1_QPS"      "$CLUSTER_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=3"   "$TOP3_RECALL"     "$TOP3_QPS"      "$CLUSTER_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=5"   "$TOP5_RECALL"     "$TOP5_QPS"      "$CLUSTER_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=10"  "$TOP10_RECALL"    "$TOP10_QPS"     "$CLUSTER_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=20"  "$TOP20_RECALL"    "$TOP20_QPS"     "$CLUSTER_BUILD"
printf "%-25s %-12s %-12s %-12s\n" "Clustered TOP=50"  "$TOP50_RECALL"    "$TOP50_QPS"     "$CLUSTER_BUILD"
echo "========================================================"
echo "* ANN-benchmarks: Linux server, hnswlib ef=50, M=16"

# ── Save to CSV ───────────────────────────────────────────────
# Write header if file doesn't exist
if [ ! -f "$RESULTS_CSV" ]; then
  echo "timestamp,machine,threads,method,recall,qps,build_time" > "$RESULTS_CSV"
fi

# Append results
echo "$TIMESTAMP,$MACHINE,$THREADS,baseline,$BASELINE_RECALL,$BASELINE_QPS,$BASELINE_BUILD"          >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,ann_benchmarks,0.9500,28021,N/A"                                   >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top1,$TOP1_RECALL,$TOP1_QPS,$CLUSTER_BUILD"              >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top3,$TOP3_RECALL,$TOP3_QPS,$CLUSTER_BUILD"              >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top5,$TOP5_RECALL,$TOP5_QPS,$CLUSTER_BUILD"              >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top10,$TOP10_RECALL,$TOP10_QPS,$CLUSTER_BUILD"           >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top20,$TOP20_RECALL,$TOP20_QPS,$CLUSTER_BUILD"           >> "$RESULTS_CSV"
echo "$TIMESTAMP,$MACHINE,$THREADS,clustered_top50,$TOP50_RECALL,$TOP50_QPS,$CLUSTER_BUILD"           >> "$RESULTS_CSV"

echo ""
echo "Results saved to: $RESULTS_CSV"
echo "Open dashboard:   open ../../dashboard.html"
