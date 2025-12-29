import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ===========================
# 1. Figure
# ===========================
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")

# ===========================
# 2. Geometry
# ===========================
S = np.array([-5.0, 3.0, -8.0])   # X-ray source

det_z = 10.0
det_size = 15.0
n_det = 8
pixel_size = det_size / n_det
d = det_size / 2

# Detector pixel (u,v) center
u_idx, v_idx = 5, 2
u_center = -d + (u_idx + 0.5) * pixel_size
v_center = -d + (v_idx + 0.5) * pixel_size
D = np.array([u_center, v_center, det_z])

# CT volume
cube_size = 6.0
half = cube_size / 2
n_vox = 6
voxel_size = cube_size / n_vox

# ===========================
# 3. Source and detector pixel
# ===========================
ax.scatter(*S, color="red", s=120)
ax.text(*(S + [1.5, -1.5, 1.5]), "S", color="red", fontsize=14)

ax.scatter(*D, color="darkgreen", s=90)
ax.text(*(D + [1.5, -1.5, -1.5]), "D(u,v)", color="darkgreen", fontsize=14)

# ===========================
# 4. Detector plane & pixel grid
# ===========================
detector_face = np.array([
    [-d, -d, det_z],
    [ d, -d, det_z],
    [ d,  d, det_z],
    [-d,  d, det_z],
])
ax.add_collection3d(
    Poly3DCollection([detector_face], facecolor="palegreen", edgecolor="green", alpha=0.25)
)

det_grid = np.linspace(-d, d, n_det + 1)
for g in det_grid:
    ax.plot([-d, d], [g, g], [det_z, det_z], color="green", lw=0.8)
    ax.plot([g, g], [-d, d], [det_z, det_z], color="green", lw=0.8)

# ===========================
# 5. Draw CT volume voxels (Background)
# ===========================
def get_voxel_faces(center, size):
    h = size / 2
    cx, cy, cz = center
    pts = np.array([
        [cx-h, cy-h, cz-h], [cx+h, cy-h, cz-h],
        [cx+h, cy+h, cz-h], [cx-h, cy+h, cz-h],
        [cx-h, cy-h, cz+h], [cx+h, cy-h, cz+h],
        [cx+h, cy+h, cz+h], [cx-h, cy+h, cz+h],
    ])
    return [
        [pts[0], pts[1], pts[2], pts[3]], [pts[4], pts[5], pts[6], pts[7]],
        [pts[0], pts[1], pts[5], pts[4]], [pts[2], pts[3], pts[7], pts[6]],
        [pts[1], pts[2], pts[6], pts[5]], [pts[4], pts[7], pts[3], pts[0]],
    ]

for i in range(n_vox):
    for j in range(n_vox):
        for k in range(n_vox):
            cx = -half + (i + 0.5) * voxel_size
            cy = -half + (j + 0.5) * voxel_size
            cz = -half + (k + 0.5) * voxel_size
            faces = get_voxel_faces([cx, cy, cz], voxel_size)
            ax.add_collection3d(Poly3DCollection(faces, facecolor="lightsteelblue", edgecolor=None, alpha=0.05))

# ===========================
# 6. Ray & Intersections (Siddon's concept)
# ===========================
ax.plot([S[0], D[0]], [S[1], D[1]], [S[2], D[2]], color="orange", linewidth=3)

direction = D - S
t_vals = [0.0, 1.0] # Start and end
grid = np.linspace(-half, half, n_vox + 1)

for axis in range(3):
    for g in grid:
        t = (g - S[axis]) / direction[axis]
        if 0 < t < 1:
            P = S + t * direction
            # Check if intersection is within the volume boundaries
            if np.all(P >= -half - 1e-9) and np.all(P <= half + 1e-9):
                t_vals.append(t)

t_vals = sorted(list(set(np.round(t_vals, 8))))

# ===========================
# 7. Highlight traversed voxels
# ===========================
centers = []
for i in range(len(t_vals)-1):
    t_mid = (t_vals[i] + t_vals[i+1]) / 2
    p_mid = S + t_mid * direction
    # Find voxel index for the midpoint of the segment
    idx = np.floor((p_mid + half) / voxel_size)
    idx = np.clip(idx, 0, n_vox - 1)
    c = -half + (idx + 0.5) * voxel_size
    
    # Only draw if the segment is actually inside the volume
    if np.all(p_mid >= -half) and np.all(p_mid <= half):
        if not any(np.allclose(c, cc) for cc in centers):
            centers.append(c)

for c in centers:
    faces = get_voxel_faces(c, voxel_size)
    ax.add_collection3d(Poly3DCollection(faces, facecolor="firebrick", edgecolor="black", lw=0.5, alpha=0.4))

# ===========================
# 8. 关键修改：设置等比例显示
# ===========================
# 1. 强制 1:1:1 的数据长宽比
ax.set_box_aspect([1, 1, 1]) 

# 2. 设置三个轴的范围一致（范围需覆盖所有物体：Source在-14，Detector在+14）
max_range = 16 
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# ===========================
# 9. View
# ===========================
ax.axis("off")
ax.view_init(elev=20, azim=35)
plt.tight_layout()
plt.show()