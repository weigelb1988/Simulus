#!/usr/bin/env python3
"""
Quick performance diagnostic to compare training throughput.

This helps determine if the optimizations actually improved training speed
by measuring samples/second, not just iterations/second.
"""

import time

# Your reported metrics
print("=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

# Before optimizations
iter_per_sec_before = 15
batch_size_before = 32
samples_per_sec_before = iter_per_sec_before * batch_size_before

print("\n📊 BEFORE OPTIMIZATIONS:")
print(f"  Iterations/sec: {iter_per_sec_before}")
print(f"  Batch size:     {batch_size_before}")
print(f"  ➡️  Throughput:  {samples_per_sec_before} samples/sec")
print(f"  Steps per epoch: 200")
print(f"  ➡️  Time per epoch: {200 / iter_per_sec_before:.1f} seconds")

# After optimizations
iter_per_sec_after = 5
batch_size_after = 128
samples_per_sec_after = iter_per_sec_after * batch_size_after

print("\n📊 AFTER OPTIMIZATIONS:")
print(f"  Iterations/sec: {iter_per_sec_after}")
print(f"  Batch size:     {batch_size_after}")
print(f"  ➡️  Throughput:  {samples_per_sec_after} samples/sec")
print(f"  Steps per epoch: 200")
print(f"  ➡️  Time per epoch: {200 / iter_per_sec_after:.1f} seconds")

# Comparison
speedup = samples_per_sec_after / samples_per_sec_before
epoch_time_ratio = (200 / iter_per_sec_after) / (200 / iter_per_sec_before)

print("\n" + "=" * 60)
print("VERDICT:")
print("=" * 60)

if speedup > 1.0:
    print(f"✅ You're FASTER by {speedup:.2f}x in throughput!")
    print(f"✅ Epoch time improved by {1/epoch_time_ratio:.2f}x")
    print(f"\n   Before: {200 / iter_per_sec_before:.1f}s per epoch")
    print(f"   After:  {200 / iter_per_sec_after:.1f}s per epoch")
    print(f"   Saved:  {(200 / iter_per_sec_before) - (200 / iter_per_sec_after):.1f}s per epoch")
elif speedup > 0.95:
    print(f"⚠️  Roughly the same speed ({speedup:.2f}x)")
else:
    print(f"❌ You're SLOWER by {1/speedup:.2f}x")
    print("   This suggests a real performance problem!")

print("\n" + "=" * 60)
print("WHAT TO CHECK:")
print("=" * 60)
print("\n1. First epoch slowdown (torch.compile warmup)?")
print("   → First epoch is always slower due to JIT compilation")
print("   → Check if epoch 2+ are faster")

print("\n2. GPU utilization:")
print("   → Run: nvidia-smi -l 1")
print("   → Should see 70-90% GPU utilization")
print("   → If low, there's a bottleneck")

print("\n3. Mixed precision overhead:")
print("   → If GPU doesn't support FP16 well, AMP can slow things down")
print("   → Check GPU model (works best on: A100, V100, RTX 30/40 series)")

print("\n4. Memory issues:")
print("   → If batch size is too large, can cause slowdowns")
print("   → Check for OOM warnings or memory pressure")

print("\n" + "=" * 60)
