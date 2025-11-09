#!/usr/bin/env python3
"""
Detailed test: Compare two conversion methods
1. Column negation (current approach): c2w[:3, 1:3] *= -1
2. Matrix multiplication (AnySplat approach): c2w @ transform_matrix
"""

import numpy as np

print("="*80)
print("Comparing Two Conversion Methods")
print("="*80)

# Test camera: at (1, 2, 3), looking down -Z in OpenGL
c2w_opengl = np.array([
    [1,  0,  0,  1],   # X right, camera x=1
    [0,  1,  0,  2],   # Y up, camera y=2  
    [0,  0,  1,  3],   # Z back, camera z=3
    [0,  0,  0,  1]
], dtype=float)

print("\nOriginal C2W (OpenGL - Y up, Z back):")
print(c2w_opengl)
print(f"Camera position: {c2w_opengl[:3, 3]}")
print(f"Camera X-axis: {c2w_opengl[:3, 0]}")
print(f"Camera Y-axis: {c2w_opengl[:3, 1]}")
print(f"Camera Z-axis: {c2w_opengl[:3, 2]}")

# Method 1: Negate columns (current approach in our code)
print("\n" + "-"*80)
print("METHOD 1: Negate Y and Z columns")
print("-"*80)
c2w_method1 = c2w_opengl.copy()
c2w_method1[:3, 1:3] *= -1

print("\nC2W after column negation:")
print(c2w_method1)
print(f"Camera position: {c2w_method1[:3, 3]}")
print(f"Camera X-axis: {c2w_method1[:3, 0]}")
print(f"Camera Y-axis (should point down): {c2w_method1[:3, 1]}")
print(f"Camera Z-axis (should point forward): {c2w_method1[:3, 2]}")

w2c_method1 = np.linalg.inv(c2w_method1)
print(f"\nW2C translation: {w2c_method1[:3, 3]}")

# Method 2: Matrix multiplication (AnySplat approach)
print("\n" + "-"*80)
print("METHOD 2: Right-multiply by transformation matrix")
print("-"*80)
blender2opencv = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

c2w_method2 = c2w_opengl @ blender2opencv

print("\nC2W after matrix multiplication:")
print(c2w_method2)
print(f"Camera position: {c2w_method2[:3, 3]}")
print(f"Camera X-axis: {c2w_method2[:3, 0]}")
print(f"Camera Y-axis (should point down): {c2w_method2[:3, 1]}")
print(f"Camera Z-axis (should point forward): {c2w_method2[:3, 2]}")

w2c_method2 = np.linalg.inv(c2w_method2)
print(f"\nW2C translation: {w2c_method2[:3, 3]}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Camera positions match: {np.allclose(c2w_method1[:3, 3], c2w_method2[:3, 3])}")
print(f"Rotations match: {np.allclose(c2w_method1[:3, :3], c2w_method2[:3, :3])}")

if np.allclose(c2w_method1, c2w_method2):
    print("\n✅ Both methods produce IDENTICAL results!")
else:
    print("\n❌ Methods produce DIFFERENT results!")
    print("\nDifference:")
    print(c2w_method1 - c2w_method2)

# Check what makes sense
print("\n" + "="*80)
print("SEMANTIC CHECK")
print("="*80)
print("\nIn OpenGL: camera at (1, 2, 3)")
print("  - Y=2 means 2 units UP")
print("  - Z=3 means 3 units BACK (away from screen)")
print("\nIn OpenCV: camera should be at...")
print("  - X should stay 1 (X is same)")
print("  - Y should be -2 (UP becomes DOWN)")
print("  - Z should be -3 (BACK becomes FORWARD)")
print(f"\nMethod 1 gives: {c2w_method1[:3, 3]} {'❌ WRONG - position unchanged!' if c2w_method1[1, 3] == 2 else '✅'}")
print(f"Method 2 gives: {c2w_method2[:3, 3]} {'✅ CORRECT - Y and Z flipped!' if c2w_method2[1, 3] == 2 else '❌'}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if np.allclose(c2w_method1, c2w_method2):
    print("✅ Both methods are equivalent")
else:
    print("⚠️  Methods are DIFFERENT!")
    print("   Column negation does NOT transform camera position")
    print("   Matrix multiplication DOES transform camera position")
    print("\n   For coordinate system transformation, we need Method 2!")

