"""
Shearing Tutorial Utilities for JAX-MD

This module contains visualization functions and helper utilities for the
Brownian Dynamics with Shearing tutorial. It provides comprehensive tools
for understanding and visualizing shear deformation, coordinate transformations,
and simulation results.

The module includes:
- 2D and 3D box visualization functions
- Coordinate transformation demonstrations
- Animation utilities for shear protocols
- Stress and trajectory analysis tools
- Multi-plane shear visualization
- Publication-quality plotting functions

Author: JAX-MD Team
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
from collections import defaultdict

import jax_md.space as space


# =============================================================================
# Core Visualization Functions
# =============================================================================

def corners_of_box(H: jnp.ndarray) -> np.ndarray:
    """Return the corners of the unit square/cube under transform H.
    
    Args:
        H: Box matrix transformation. Shape (2,2) for 2D or (3,3) for 3D.
        
    Returns:
        Array of corner coordinates after transformation.
    """
    if H.shape[0] == 2:
        corners = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],  # Close the polygon
        ])
    else:  # 3D
        corners = jnp.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
        ])
    P = space.transform(H, corners)
    return np.asarray(P)


def grid_lines_2d(H: jnp.ndarray, n: int = 8, m: int = 50) -> List[np.ndarray]:
    """Generate deformed grid lines in 2D real space under transform H.
    
    Args:
        H: 2D box matrix transformation.
        n: Number of grid lines in each direction.
        m: Number of points per grid line.
        
    Returns:
        List of arrays containing grid line coordinates.
    """
    us = jnp.linspace(0.0, 1.0, n)
    vs = jnp.linspace(0.0, 1.0, n)
    ts = jnp.linspace(0.0, 1.0, m)
    lines: List[np.ndarray] = []
    
    # Vertical lines: (u, t)
    for u in us:
        line = jnp.stack([jnp.full_like(ts, u), ts], axis=1)
        real_line = space.transform(H, line)
        lines.append(np.asarray(real_line))
    
    # Horizontal lines: (t, v)
    for v in vs:
        line = jnp.stack([ts, jnp.full_like(ts, v)], axis=1)
        real_line = space.transform(H, line)
        lines.append(np.asarray(real_line))
    
    return lines


def plot_box_and_grid_2d(H: jnp.ndarray, ax=None, title: str = "", 
                        show_grid: bool = True, show_fractional: bool = False, 
                        grid_alpha: float = 0.3, box_color: str = 'blue', xlim=None, ylim=None):
    """Plot 2D box with deformed grid.
    
    Args:
        H: 2D box matrix transformation.
        ax: Matplotlib axes object. If None, creates new figure.
        title: Title for the plot.
        show_grid: Whether to show deformed grid lines.
        show_fractional: Whether to show unit square reference.
        grid_alpha: Transparency of grid lines.
        box_color: Color of the box boundary.
        
    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot the box boundary
    corners = corners_of_box(H)
    ax.plot(corners[:, 0], corners[:, 1], '-', linewidth=3, color=box_color)
    
    # Plot deformed grid
    if show_grid:
        lines = grid_lines_2d(H, n=6)
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], '-', alpha=grid_alpha, color='gray', linewidth=1)
    
    # Plot fractional coordinate reference
    if show_fractional:
        # Unit square in light color
        unit_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        ax.plot(unit_corners[:, 0], unit_corners[:, 1], '--', 
                alpha=0.5, color='red', linewidth=2, label='Unit square')

    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if title and show_fractional:
        ax.legend()
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Make x and y axes equal
    ax.set_aspect('equal')
    
    return ax


def visualize_coordinate_transformation(R_frac: jnp.ndarray, H: jnp.ndarray, 
                                      title_prefix: str = ""):
    """Visualize particles in both fractional and real coordinates.
    
    Args:
        R_frac: Particle positions in fractional coordinates.
        H: Box matrix transformation.
        title_prefix: Prefix for plot titles.
        
    Returns:
        Tuple of (figure, (ax1, ax2)) where ax1 is fractional and ax2 is real coordinates.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Fractional coordinates (left)
    ax1.scatter(R_frac[:, 0], R_frac[:, 1], s=100, alpha=0.7, c='red', 
                edgecolors='darkred', linewidth=2, label='Particles')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title_prefix}Fractional Coordinates', fontsize=14, fontweight='bold')
    ax1.set_xlabel('s_x (fractional)')
    ax1.set_ylabel('s_y (fractional)')
    
    # Unit square boundary
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax1.plot(unit_square[:, 0], unit_square[:, 1], 'k-', linewidth=2, label='Unit cell')
    ax1.legend()
    
    # Real coordinates (right)
    R_real = space.transform(H, R_frac)
    ax2 = plot_box_and_grid_2d(H, ax=ax2, title=f'{title_prefix}Real Coordinates', 
                              show_grid=True, box_color='blue')
    ax2.scatter(R_real[:, 0], R_real[:, 1], s=100, alpha=0.7, c='red',
                edgecolors='darkred', linewidth=2, label='Particles')
    ax2.set_xlabel('x (real)')
    ax2.set_ylabel('y (real)')
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# 3D Visualization Functions
# =============================================================================

def corners_of_box_3d(H: jnp.ndarray) -> np.ndarray:
    """Return the 8 corners of the 3D unit cube under transform H.
    
    Args:
        H: 3D box matrix transformation.
        
    Returns:
        Array of 8 corner coordinates after transformation.
    """
    corners = jnp.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])
    P = space.transform(H, corners)
    return np.asarray(P)


def box_edges_indices_3d() -> List[Tuple[int, int]]:
    """Indices of edges connecting the 8 cube corners.
    
    Returns:
        List of tuples representing edge connections.
    """
    return [
        # Bottom face edges
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face edges  
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]


def plot_3d_box(H: jnp.ndarray, ax: Any = None, color: str = 'blue', alpha: float = 0.3, 
                show_edges: bool = True, show_faces: bool = True):
    """Plot 3D sheared box with faces and edges.
    
    Args:
        H: 3D box matrix transformation.
        ax: 3D matplotlib axes. If None, creates new figure.
        color: Face color for the box.
        alpha: Transparency of faces.
        show_edges: Whether to show box edges.
        show_faces: Whether to show box faces.
        
    Returns:
        3D matplotlib axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    corners = corners_of_box_3d(H)
    
    if show_faces:
        # Define the 6 faces of the cube
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[3], corners[7], corners[4]],  # left
            [corners[1], corners[2], corners[6], corners[5]]   # right
        ]
        
        face_collection = Poly3DCollection(faces, alpha=alpha, 
                                         facecolors=color, edgecolors='black')
        ax.add_collection3d(face_collection)
    
    if show_edges:
        edges = box_edges_indices_3d()
        for i, j in edges:
            ax.plot([corners[i, 0], corners[j, 0]], 
                   [corners[i, 1], corners[j, 1]], 
                   [corners[i, 2], corners[j, 2]], 
                   'k-', linewidth=2)
    
    # Set equal aspect ratio
    max_range = np.array([corners[:, 0].max() - corners[:, 0].min(),
                         corners[:, 1].max() - corners[:, 1].min(),
                         corners[:, 2].max() - corners[:, 2].min()]).max() / 2.0
    
    mid_x = (corners[:, 0].max() + corners[:, 0].min()) * 0.5
    mid_y = (corners[:, 1].max() + corners[:, 1].min()) * 0.5
    mid_z = (corners[:, 2].max() + corners[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def visualize_multi_plane_shear(gamma_xy: float = 0.3, gamma_xz: float = 0.2, 
                               gamma_yz: float = 0.1):
    """Visualize the effect of multi-plane shearing.
    
    Args:
        gamma_xy: Shear strain in xy plane.
        gamma_xz: Shear strain in xz plane.
        gamma_yz: Shear strain in yz plane.
        
    Returns:
        Tuple of (figure, (ax1, ax2), H_base, H_sheared).
    """
    # Base cubic box
    H_base = jnp.eye(3) * 2.0  # 2x2x2 cube
    
    # Apply multi-plane shear
    H_sheared = H_base.copy()
    H_sheared = H_sheared.at[0, 1].set(gamma_xy * H_base[1, 1])  # xy shear
    H_sheared = H_sheared.at[0, 2].set(gamma_xz * H_base[2, 2])  # xz shear
    H_sheared = H_sheared.at[1, 2].set(gamma_yz * H_base[2, 2])  # yz shear
    
    fig = plt.figure(figsize=(16, 8))
    
    # Original box
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_box(H_base, ax=ax1, color='lightblue', alpha=0.3)
    ax1.set_title('Original Cubic Box', fontsize=14, fontweight='bold')
    
    # Sheared box
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_box(H_sheared, ax=ax2, color='lightcoral', alpha=0.3)
    ax2.set_title(f'Multi-Plane Sheared Box\nγ_xy={gamma_xy}, γ_xz={gamma_xz}, γ_yz={gamma_yz}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, (ax1, ax2), H_base, H_sheared


# =============================================================================
# Animation and Dynamic Visualization
# =============================================================================

def create_shear_animation(shear_fn: Callable, duration: float = 4.0, 
                          fps: int = 20, title: str = "Shear Animation"):
    """Create an animation showing how the box deforms under shear.
    
    Args:
        shear_fn: Function that takes time and returns shear strain.
        duration: Duration of animation in seconds.
        fps: Frames per second for animation.
        title: Title for the animation.
        
    Returns:
        Tuple of (animation, figure).
    """
    times = np.linspace(0, duration, int(duration * fps))
    
    # Precompute all gamma values to determine fixed plot limits
    gamma_values = [shear_fn(t) for t in times]
    gamma_max = np.max(gamma_values)
    gamma_min = np.min(gamma_values)
    y_margin = 0.15 * max(abs(gamma_max), abs(gamma_min))
    
    # Make figure wider with consistent styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Fixed axis limits for consistent box size
    box_size = 2.0  # Base size of box
    max_shear_width = box_size * abs(gamma_max)
    
    # Make room at the bottom of the figure for text
    fig.subplots_adjust(bottom=0.15)  # Make room for the progress indicator
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        t = times[frame]
        gamma = shear_fn(t)
        
        # Normalize current time for color mapping and progress
        t_norm = frame / (len(times) - 1)
        
        # Use simple color transitions instead of colormap
        current_color = 'royalblue' if gamma >= 0 else 'crimson'
        
        # Create sheared box matrix with fixed height
        H = jnp.array([[box_size, gamma * box_size], [0.0, box_size]])
        
        # Calculate proper xlims accounting for both positive and negative shear
        # The sheared box can extend in either direction depending on gamma sign
        shear_offset = gamma * box_size
        if gamma >= 0:
            xmin = -0.5
            xmax = box_size + max_shear_width + 0.5
        else:
            xmin = min(-0.5, shear_offset - 0.5)
            xmax = box_size + 0.5
            
        # Ensure we can always see the full range of motion
        xmin = min(-0.5, -max_shear_width - 0.5)
        xmax = max(box_size + 0.5, box_size + max_shear_width + 0.5)
        
        ymin = -0.5
        ymax = box_size + 0.5
        
        # First plot the original (unsheared) box outline as reference
        H_original = jnp.array([[box_size, 0.0], [0.0, box_size]])
        corners_original = corners_of_box(H_original)
        ax1.plot(corners_original[:, 0], corners_original[:, 1], '--', 
                linewidth=2, color='gray', alpha=0.6, label='Original box')
        
        # Plot the sheared box with enhanced styling
        plot_box_and_grid_2d(H, ax=ax1, title=f'Sheared Box at t={t:.2f}s', 
                           show_grid=True, box_color=current_color, 
                           xlim=(xmin, xmax), ylim=(ymin, ymax),
                           grid_alpha=0.2)
        
        # Add shear direction arrows
        if abs(gamma) > 0.01:  # Only show arrows when there's significant shear
            arrow_dir = 1 if gamma > 0 else -1
            arrow_x = box_size / 2
            arrow_y = box_size
            dx = gamma * box_size / 2  # Half the shear displacement
            
            # Top arrow
            ax1.arrow(arrow_x, arrow_y, dx, 0, 
                     head_width=0.15, head_length=0.1, 
                     fc=current_color, ec=current_color, linewidth=2)
            
            # Add γ label
            ax1.text(arrow_x + dx/2, arrow_y + 0.3, f'γ = {gamma:.2f}', 
                    ha='center', va='center', fontsize=14, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Add a small legend in the top right corner
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.8)
        
        # Plot the shear function with enhanced styling
        ax2.plot(times[:frame+1], gamma_values[:frame+1], '-', 
               linewidth=3, color='royalblue', label='γ(t)')
        
        # Fill between curve and zero
        ax2.fill_between(times[:frame+1], gamma_values[:frame+1], 
                        alpha=0.2, color='royalblue')
        
        # Current point with larger marker
        ax2.scatter(t, gamma, color=current_color, s=120, 
                  edgecolor='black', linewidth=1.5, zorder=5)
        
        # Set fixed limits for consistent view
        ax2.set_xlim(0, duration)
        ax2.set_ylim(gamma_min - y_margin, gamma_max + y_margin)
        
        # Enhanced styling
        ax2.set_xlabel('Time', fontsize=14)
        ax2.set_ylabel('Shear Strain γ(t)', fontsize=14)
        ax2.set_title('Shear Function Evolution', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add vertical line at current time
        ax2.axvline(t, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        # Add completion percentage
        # fig.text(0.5, 0.08, f"Progress: {t_norm*100:.0f}%", ha='center', fontsize=12)
        
        # Return a list of artists to make FuncAnimation happy
        return [ax1, ax2]
    
    anim = FuncAnimation(fig, animate, frames=len(times), interval=1000/fps, repeat=True)
    
    # Apply tight layout but preserve our custom spacing
    plt.tight_layout(rect=(0, 0.2, 1, 0.95))
    
    return anim, fig


def unwrap_trajectory(traj_frac: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    """Unwrap trajectory to handle periodic boundary crossings.
    
    Args:
        traj_frac: Trajectory in fractional coordinates. Shape (n_frames, 2).
        
    Returns:
        Unwrapped trajectory where periodic jumps are corrected.
    """
    # Convert to numpy array for processing
    traj = np.asarray(traj_frac)
    
    if len(traj) <= 1:
        return traj.copy()
    
    unwrapped = np.zeros_like(traj)
    unwrapped[0] = traj[0]
    
    for i in range(1, len(traj)):
        diff = traj[i] - traj[i-1]
        # Detect periodic jumps (threshold of 0.5 in fractional coordinates)
        jump = np.round(diff)
        unwrapped[i] = unwrapped[i-1] + (diff - jump)
    
    return unwrapped


def create_trajectory_animation(trajectory: jnp.ndarray, box_history: jnp.ndarray, 
                               times: jnp.ndarray, shear_fn: Callable, 
                               fps: int = 10, title: str = "Particle Trajectories"):
    """Create an animation showing particle trajectories in the xy plane with periodic images.
    
    Args:
        trajectory: Array of particle positions over time. Shape (n_frames, n_particles, n_dims).
        box_history: Array of box matrices over time. Shape (n_frames, n_dims, n_dims).  
        times: Array of time values. Shape (n_frames,).
        shear_fn: Function that takes time and returns shear strain.
        fps: Frames per second for animation.
        title: Title for the animation.
        
    Returns:
        Tuple of (animation, figure).
    """
    N_particles = trajectory.shape[1]
    
    # Create figure with consistent styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.3)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Set up the plots with consistent styling - extend view to show phantom images
    ax1.set_xlim(-1.1, 2.1)
    ax1.set_ylim(-1.1, 2.1)
    ax1.set_aspect('equal')
    ax1.set_title('Fractional Coordinates + Images (xy plane)', fontsize=14, fontweight='bold')
    ax1.grid(False)
    
    # Unit square boundary for fractional coordinates (main and periodic images)
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax1.plot(unit_square[:, 0], unit_square[:, 1], 'k-', linewidth=2, label='Main cell')
    
    # Draw periodic image boundaries (fainter)
    for i in [-1, 0, 1, 2]:
        for j in [-1, 0, 1, 2]:
            if i != 0 or j != 0:  # Skip the main unit cell
                image_square = unit_square + np.array([i, j])
                ax1.plot(image_square[:, 0], image_square[:, 1], 'k--', 
                        linewidth=1, alpha=0.3)
    
    # Real coordinates setup - we'll update the limits dynamically
    ax2.set_aspect('equal')
    ax2.set_title('Real Coordinates (xy plane)', fontsize=14, fontweight='bold')
    ax2.grid(False)
    
    # Initialize empty plots for main particles
    scat1 = ax1.scatter([], [], s=50, alpha=0.7, c='red', edgecolors='darkred', linewidth=1)
    scat2 = ax2.scatter([], [], s=50, alpha=0.7, c='red', edgecolors='darkred', linewidth=1)
    
    # Initialize empty plots for phantom particles (periodic images)
    phantom_scats1 = []  # Fractional phantoms
    phantom_scats2 = []  # Real phantoms
    
    # Create phantom scatters for 8 neighboring periodic images (2D)
    image_offsets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i != 0 or j != 0:  # Skip the main image
                image_offsets.append([i, j])
                # Fractional phantom (fainter)
                phantom1 = ax1.scatter([], [], s=30, alpha=0.2, c='red', 
                                     edgecolors='darkred', linewidth=0.5)
                phantom_scats1.append(phantom1)
                # Real phantom (fainter)
                phantom2 = ax2.scatter([], [], s=30, alpha=0.2, c='red', 
                                     edgecolors='darkred', linewidth=0.5)
                phantom_scats2.append(phantom2)
    
    # Box lines for real coordinates (main and phantom boxes)
    box_lines = ax2.plot([], [], 'b-', linewidth=2)[0]
    
    # Initialize phantom box lines
    phantom_box_lines = []
    for i, offset in enumerate(image_offsets):
        phantom_box = ax2.plot([], [], 'b--', linewidth=1, alpha=0.3)[0]
        phantom_box_lines.append(phantom_box)
    
    # Compute global x/y limits across all frames and periodic images to account for remapping
    # We use the actual box history (including remaps) and the same periodic images we will render.
    all_offsets = [[0, 0]] + image_offsets
    x_min_global, x_max_global = np.inf, -np.inf
    y_min_global, y_max_global = np.inf, -np.inf
    n_frames_total = int(box_history.shape[0])
    for f in range(n_frames_total):
        Hf = box_history[f, :2, :2]
        main_corners = corners_of_box(Hf)  # np array of polygon corners (closed)
        for off in all_offsets:
            lattice_shift = np.asarray(Hf) @ np.array(off)
            corners_shifted = main_corners + lattice_shift
            x_min_global = min(x_min_global, float(corners_shifted[:, 0].min()))
            x_max_global = max(x_max_global, float(corners_shifted[:, 0].max()))
            y_min_global = min(y_min_global, float(corners_shifted[:, 1].min()))
            y_max_global = max(y_max_global, float(corners_shifted[:, 1].max()))

    # Add small margins for visuals
    x_range = x_max_global - x_min_global
    y_range = y_max_global - y_min_global
    x_margin = 0.05 * x_range if np.isfinite(x_range) and x_range > 0 else 0.5
    y_margin = 0.05 * y_range if np.isfinite(y_range) and y_range > 0 else 0.5
    ax2.set_xlim(x_min_global - x_margin, x_max_global + x_margin)
    ax2.set_ylim(y_min_global - y_margin, y_max_global + y_margin)

    # Text displays
    # Place text in the bottom-left figure margin to avoid overlapping subplot titles
    time_text = fig.text(0.02, 0.06, '', fontsize=12, fontweight='bold')
    gamma_text = fig.text(0.02, 0.02, '', fontsize=12, fontweight='bold')
    
    # Trajectory lines (sample a few particles for clarity)
    sample_particles = range(0, N_particles, max(1, N_particles//8))
    # Use predefined colors for trajectory trails
    available_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    colors = [available_colors[i % len(available_colors)] for i in range(len(sample_particles))]
    
    trail_lines1 = []  # Fractional coordinate trails
    trail_lines2 = []  # Real coordinate trails
    
    for i, color in enumerate(colors):
        line1, = ax1.plot([], [], '-', color=color, alpha=0.7, linewidth=3)
        line2, = ax2.plot([], [], '-', color=color, alpha=0.7, linewidth=3)
        trail_lines1.append(line1)
        trail_lines2.append(line2)
    
    def animate(frame):
        # Get current time step data
        t = times[frame]
        pos_frac = trajectory[frame, :, :2]  # Only xy coordinates
        H = box_history[frame, :2, :2]  # Only 2D box matrix
        
        # Transform to real coordinates
        pos_real = space.transform(H, trajectory[frame, :, :2])
        
        # Update main particle positions
        scat1.set_offsets(pos_frac)
        scat2.set_offsets(pos_real)
        
        # Update phantom particles and boxes (periodic images)
        for i, offset in enumerate(image_offsets):
            # Fractional phantom positions in neighboring images
            # (do not wrap back into unit cell so we can show distinct images in fractional view)
            offset_vec = np.array(offset)
            phantom_frac = pos_frac + offset_vec
            phantom_scats1[i].set_offsets(phantom_frac)

            # Real phantom positions: r_image = H * (s + offset) = H*s + H*offset
            # Compute the lattice shift once and translate current real positions
            lattice_shift = H @ offset_vec
            phantom_real = pos_real + lattice_shift
            phantom_scats2[i].set_offsets(phantom_real)

            # Update phantom box boundaries in real coordinates by the same lattice shift
            main_corners = corners_of_box(H)
            phantom_corners_shifted = main_corners + lattice_shift
            phantom_box_lines[i].set_data(phantom_corners_shifted[:, 0], phantom_corners_shifted[:, 1])
        
        # Update box in real coordinates - draw main box and phantom boxes
        corners = corners_of_box(H)
        box_lines.set_data(corners[:, 0], corners[:, 1])
        
    # Axes limits are fixed globally above (accounting for remapping); no per-frame updates.
        
        # Update trajectory trails (show last 20 frames) with periodic unwrapping
        trail_length = min(20, frame + 1)
        start_frame = max(0, frame - trail_length + 1)
        
        for i, p_idx in enumerate(sample_particles):
            # Fractional coordinate trails - unwrap periodic jumps
            traj_frac_raw = trajectory[start_frame:frame+1, p_idx, :2]
            traj_frac_unwrapped = unwrap_trajectory(traj_frac_raw)
            trail_lines1[i].set_data(traj_frac_unwrapped[:, 0], traj_frac_unwrapped[:, 1])
            
            # Real coordinate trails - transform unwrapped fractional coordinates
            # This creates continuous trajectories that don't jump across boundaries
            traj_real = []
            for j in range(len(traj_frac_unwrapped)):
                H_j = box_history[start_frame + j, :2, :2]
                # Use unwrapped fractional coordinates (can be outside [0,1))
                pos_real_j = space.transform(H_j, traj_frac_unwrapped[j])
                traj_real.append(pos_real_j)
            
            if traj_real:
                traj_real = np.array(traj_real)
                trail_lines2[i].set_data(traj_real[:, 0], traj_real[:, 1])
        
        # Update text
        gamma_val = shear_fn(t)
        # Keep to 3 decimals for cleaner display
        time_text.set_text(f'Time: {t:.3f}')
        gamma_text.set_text(f'Shear strain γ: {gamma_val:.3f}')
        
        return ([scat1, scat2, box_lines, time_text, gamma_text] + 
                phantom_scats1 + phantom_scats2 + phantom_box_lines + 
                trail_lines1 + trail_lines2)
    
    # Create animation - limit frames for performance
    frames_to_show = min(len(times), 100)
    frame_indices = np.linspace(0, len(times)-1, frames_to_show, dtype=int)
    
    anim = FuncAnimation(fig, animate, frames=frame_indices, interval=1000/fps, 
                        blit=False, repeat=True)
    
    # Apply tight layout
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))
    
    return anim, fig


def plot_static_configuration(
    R_frac: Union[jnp.ndarray, np.ndarray],
    H: jnp.ndarray,
    title: str = "Static Configuration",
    n_images: int = 1,
    radii_frac: Optional[Union[float, jnp.ndarray, np.ndarray]] = None,
    fancy: bool = True,
    figsize: Tuple[int, int] = (8, 8),
):
    """Plot a publication-quality static image of a configuration in real space only.

    Shows only the primary box and particles in real coordinates (no periodic images,
    no fractional coordinate panel).

    Args:
        R_frac: Particle positions in fractional coordinates, shape (N, 2) or (N, >=2).
        H: 2x2 real-space box matrix for xy-plane visualization.
        title: Figure title.
        n_images: Number of periodic image layers to show in each direction (ignored).
        radii_frac: Optional per-particle radii in fractional units (of box height). If float,
            used for all particles. If None, a sensible default is used.
        fancy: If True, draw glossy 3D-like spheres; if False, draw simple scatter.
        figsize: Matplotlib figure size.

    Returns:
        (fig, ax_real)
    """
    R_frac = np.asarray(R_frac)
    if R_frac.shape[1] > 2:
        R_frac = R_frac[:, :2]

    # Styling consistent with animation
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # Colors to match animation
    particle_color = 'red'
    particle_edge = 'darkred'
    box_color = 'blue'

    # Helper: glossy particle painter
    def _draw_glossy_particles(ax_target, P, r, base_color=particle_color, edge_color=particle_edge, z=5):
        # Convert color to rgb
        base_rgb = np.array(mcolors.to_rgb(base_color))
        dark_rgb = base_rgb * 0.75
        light_rgb = 1 - (1 - base_rgb) * 0.2  # blend towards white

        # Ensure per-particle radii
        if np.ndim(r) == 0:
            r_scalar = float(np.asarray(r))
            radii = np.full(len(P), r_scalar, dtype=float)
        else:
            radii = np.asarray(r, dtype=float)

        # Draw per particle
        for (x, y), rr in zip(P, radii):
            # Concentric rings to fake radial gradient
            rings = 6
            for k in range(rings):
                t = k / (rings - 1)
                c = (1 - t) * dark_rgb + t * light_rgb
                alpha = 0.95 if k == 0 else 0.9
                circ = patches.Circle((x, y), rr * (1 - 0.12*k),
                                      facecolor=(*c, alpha), edgecolor=edge_color,
                                      linewidth=0.5 if k == 0 else 0.0, zorder=z+1)
                ax_target.add_patch(circ)

    # Determine radii in fractional units
    # Default ~3% of box height in fractional coords
    if radii_frac is None:
        radius_frac_val: Union[float, np.ndarray] = 0.03
    else:
        # Normalize to numpy float/array for downstream use
        if np.ndim(radii_frac) == 0:
            radius_frac_val = float(np.asarray(radii_frac))
        else:
            radius_frac_val = np.asarray(radii_frac, dtype=float)

    # Real coordinates only - primary box only
    ax.set_aspect('equal')
    ax.set_title('Real Coordinates (xy plane)', fontsize=14, fontweight='bold')
    ax.grid(False)

    # Box boundary (main box only, no periodic images)
    H2_jax = jnp.asarray(H[:2, :2])
    main_corners = corners_of_box(H2_jax)
    ax.plot(main_corners[:, 0], main_corners[:, 1], '-', color=box_color, linewidth=2)

    # Plot particles in real space (main box only)
    R_real = space.transform(H2_jax, jnp.asarray(R_frac))

    # Map fractional radius to real-space length scale
    H2_np = np.asarray(H2_jax)
    Lx = float(H2_np[0, 0])
    Ly = float(H2_np[1, 1])
    length_scale = min(abs(Lx), abs(Ly))
    if np.ndim(radius_frac_val) == 0:
        r_real = float(np.asarray(radius_frac_val)) * length_scale
    else:
        r_real = np.asarray(radius_frac_val, dtype=float) * length_scale

    if fancy:
        _draw_glossy_particles(ax, R_real, r_real)
    else:
        ax.scatter(R_real[:, 0], R_real[:, 1], s=50, alpha=0.8,
                   c=particle_color, edgecolors=particle_edge, linewidth=1)

    # Set limits in real space to include main box with a margin
    x_min, x_max = np.min(main_corners[:, 0]), np.max(main_corners[:, 0])
    y_min, y_max = np.min(main_corners[:, 1]), np.max(main_corners[:, 1])
    x_margin = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout(rect=(0, 0.0, 1, 0.95))
    return fig, ax


# =============================================================================
# Stress and Trajectory Analysis
# =============================================================================

def plot_stress_evolution(times: jnp.ndarray, stress_xy: jnp.ndarray, 
                         title: str = "Stress Evolution"):
    """Plot the evolution of shear stress over time.
    
    Args:
        times: Array of time values.
        stress_xy: Array of shear stress values.
        title: Title for the plot.
        
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(times, stress_xy, 'b-', linewidth=2, label='σ_xy')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Shear Stress σ_xy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax


def plot_particle_trajectories(positions: jnp.ndarray, box_fn: Callable, 
                              times: jnp.ndarray, every_n: int = 10):
    """Plot particle trajectories in both fractional and real coordinates.
    
    Args:
        positions: Array of particle positions over time. Shape (n_times, n_particles, ndim).
        box_fn: Function that returns box matrix at given time.
        times: Array of time values.
        every_n: Plot every n-th trajectory point.
        
    Returns:
        Tuple of (figure, (ax1, ax2)).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    n_particles = positions.shape[1]
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, n_particles))
    
    # Fractional coordinates
    for i in range(n_particles):
        traj_frac = positions[::every_n, i, :]
        ax1.plot(traj_frac[:, 0], traj_frac[:, 1], 'o-', color=colors[i], 
                alpha=0.7, markersize=3, linewidth=1)
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title('Particle Trajectories (Fractional)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('s_x')
    ax1.set_ylabel('s_y')
    ax1.grid(True, alpha=0.3)
    
    # Real coordinates
    for i in range(n_particles):
        traj_real = []
        for j in range(0, len(positions), every_n):
            H = box_fn(t=times[j])
            pos_real = space.transform(H, positions[j, i, :])
            traj_real.append(pos_real)
        traj_real = np.array(traj_real)
        ax2.plot(traj_real[:, 0], traj_real[:, 1], 'o-', color=colors[i], 
                alpha=0.7, markersize=3, linewidth=1)
    
    # Draw final box
    H_final = box_fn(t=times[-1])
    plot_box_and_grid_2d(H_final, ax=ax2, title='Particle Trajectories (Real)', 
                        show_grid=True, box_color='blue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# Shear Protocol Definitions
# =============================================================================

def constant_shear_rate(shear_rate: float) -> Callable:
    """Constant shear rate: γ(t) = γ̇ * t
    
    Args:
        shear_rate: Constant shear rate.
        
    Returns:
        Function that takes time and returns shear strain.
    """
    return lambda t: shear_rate * t


def oscillatory_shear(amplitude: float, frequency: float) -> Callable:
    """Oscillatory shear: γ(t) = γ₀ * sin(ω * t)
    
    Args:
        amplitude: Shear strain amplitude.
        frequency: Oscillation frequency.
        
    Returns:
        Function that takes time and returns shear strain.
    """
    return lambda t: amplitude * jnp.sin(2 * jnp.pi * frequency * t)


def step_strain(strain_amplitude: float, step_time: float) -> Callable:
    """Step strain: sudden jump at t = step_time
    
    Args:
        strain_amplitude: Magnitude of the step strain.
        step_time: Time at which step occurs.
        
    Returns:
        Function that takes time and returns shear strain.
    """
    return lambda t: strain_amplitude * (t >= step_time).astype(float)


def ramp_strain(strain_rate: float, duration: float) -> Callable:
    """Ramp strain with plateau: linear increase then constant
    
    Args:
        strain_rate: Rate of strain increase during ramp.
        duration: Duration of ramp phase.
        
    Returns:
        Function that takes time and returns shear strain.
    """
    return lambda t: jnp.where(t <= duration, strain_rate * t, strain_rate * duration)


# =============================================================================
# Periodic Boundary Visualization
# =============================================================================

def show_periodic_images(gamma: float, R_boundary: jnp.ndarray, 
                        n_images: int = 2) -> Tuple:
    """Show the main box and its periodic images under shear.
    
    Args:
        gamma: Shear strain value.
        R_boundary: Particle positions near boundaries.
        n_images: Number of periodic images to show in each direction.
        
    Returns:
        Tuple of (figure, axes).
    """
    H = jnp.array([[2.0, gamma * 2.0], [0.0, 2.0]])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Colors for different periodic images
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Draw main box and several periodic images
    for i in range(-n_images, n_images + 1):
        for j in range(-n_images, n_images + 1):
            # Shift for this periodic image
            shift = jnp.array([i, j])
            
            # Draw box boundary
            corners = corners_of_box(H)
            corners_shifted = corners + space.transform(H, shift)
            
            alpha = 1.0 if (i == 0 and j == 0) else 0.3
            linewidth = 3 if (i == 0 and j == 0) else 1
            color = 'black' if (i == 0 and j == 0) else 'gray'
            
            ax.plot(corners_shifted[:, 0], corners_shifted[:, 1], 
                   color=color, linewidth=linewidth, alpha=alpha)
            
            # Draw particles in this image
            R_real = space.transform(H, R_boundary + shift)
            color_idx = (i + n_images) % len(colors)
            ax.scatter(R_real[:, 0], R_real[:, 1], s=100, 
                      c=colors[color_idx], alpha=alpha, 
                      edgecolors='black', linewidth=1)
    
    # Highlight the main box
    corners_main = corners_of_box(H)
    ax.plot(corners_main[:, 0], corners_main[:, 1], 'k-', linewidth=4, label='Main box')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Periodic Images Under Shear (γ={gamma})', fontsize=14, fontweight='bold')
    ax.set_xlabel('x (real)')
    ax.set_ylabel('y (real)')
    ax.legend()
    
    return fig, ax


# =============================================================================
# Comprehensive Analysis Functions
# =============================================================================

def compare_shear_protocols(protocols: Dict[str, Callable], duration: float = 4.0):
    """Compare stress response for different shear protocols.
    
    Args:
        protocols: Dictionary mapping protocol names to shear functions.
        duration: Time duration for comparison.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, len(protocols), figsize=(6*len(protocols), 12))
    if len(protocols) == 1:
        axes = axes.reshape(-1, 1)
    
    # Time range for comparison
    t_range = jnp.linspace(0, duration, 200)
    
    for i, (name, shear_fn) in enumerate(protocols.items()):
        # Top row: shear function
        ax_shear = axes[0, i]
        gamma_vals = [shear_fn(t) for t in t_range]
        ax_shear.plot(t_range, gamma_vals, 'b-', linewidth=3)
        ax_shear.set_title(f'{name}\nShear Function γ(t)', fontsize=14, fontweight='bold')
        ax_shear.set_xlabel('Time')
        ax_shear.set_ylabel('γ(t)')
        ax_shear.grid(True, alpha=0.3)
        
        # Bottom row: box deformation animation frames
        ax_box = axes[1, i]
        
        # Show box at several times
        times_demo = [0.5, 1.5, 2.5, 3.5]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        for j, t in enumerate(times_demo):
            gamma = shear_fn(t)
            H = jnp.array([[2.0, gamma * 2.0], [0.0, 2.0]])
            corners = corners_of_box(H)
            
            # Offset boxes slightly to show evolution
            offset = j * 0.3
            corners_offset = corners + np.array([offset, 0])
            
            ax_box.fill(corners_offset[:, 0], corners_offset[:, 1], 
                       color=colors[j], alpha=0.6, 
                       label=f't={t:.1f}, γ={gamma:.2f}')
            ax_box.plot(corners_offset[:, 0], corners_offset[:, 1], 
                       'k-', linewidth=2)
        
        ax_box.set_aspect('equal')
        ax_box.set_title(f'{name}\nBox Evolution', fontsize=14, fontweight='bold')
        ax_box.legend(fontsize=8)
        ax_box.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_demonstration_grid(n_particles: int = 25) -> jnp.ndarray:
    """Create a regular grid of particles in fractional coordinates.
    
    Args:
        n_particles: Total number of particles (should be a perfect square).
        
    Returns:
        Array of particle positions in fractional coordinates.
    """
    grid_size = int(np.sqrt(n_particles))
    
    # Create a regular grid in fractional space
    x_frac = np.linspace(0.1, 0.9, grid_size)
    y_frac = np.linspace(0.1, 0.9, grid_size) 
    xx, yy = np.meshgrid(x_frac, y_frac)
    R_frac = jnp.array([xx.flatten(), yy.flatten()]).T
    
    return R_frac


def create_boundary_particles() -> jnp.ndarray:
    """Create particles near boundaries for periodic boundary demonstrations.
    
    Returns:
        Array of particle positions near box boundaries.
    """
    R_boundary = jnp.array([
        [0.05, 0.5],   # left edge
        [0.95, 0.5],   # right edge  
        [0.5, 0.05],   # bottom edge
        [0.5, 0.95],   # top edge
        [0.05, 0.05],  # bottom-left corner
        [0.95, 0.95],  # top-right corner
    ])
    return R_boundary


# =============================================================================
# Tutorial Examples and Demonstrations
# =============================================================================

def demonstrate_coordinate_transformation():
    """Demonstrate coordinate transformation with a grid of particles."""
    # Create particles
    R_frac = create_demonstration_grid(25)
    
    # Show particles at different shear strains
    shear_values = [0.0, 0.3, 0.6, 1.0]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, gamma in enumerate(shear_values):
        # Create sheared box matrix
        H = jnp.array([[2.0, gamma * 2.0], [0.0, 2.0]])
        
        # Fractional coordinates (top row) - these never change!
        ax_frac = axes[0, i]
        ax_frac.scatter(R_frac[:, 0], R_frac[:, 1], s=100, alpha=0.7, c='red', 
                       edgecolors='darkred', linewidth=2)
        ax_frac.set_xlim(-0.05, 1.05)
        ax_frac.set_ylim(-0.05, 1.05)
        ax_frac.set_aspect('equal')
        ax_frac.grid(True, alpha=0.3)
        ax_frac.set_title(f'Fractional (γ={gamma})', fontsize=12, fontweight='bold')
        if i == 0:
            ax_frac.set_ylabel('Fractional\nCoordinates', fontsize=12, fontweight='bold')
        
        # Unit square boundary
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        ax_frac.plot(unit_square[:, 0], unit_square[:, 1], 'k-', linewidth=2)
        
        # Real coordinates (bottom row) - these change with shear!
        ax_real = axes[1, i]
        plot_box_and_grid_2d(H, ax=ax_real, title=f'Real (γ={gamma})', 
                            show_grid=True, box_color='blue')
        R_real = space.transform(H, R_frac)
        ax_real.scatter(R_real[:, 0], R_real[:, 1], s=100, alpha=0.7, c='red',
                       edgecolors='darkred', linewidth=2)
        if i == 0:
            ax_real.set_ylabel('Real\nCoordinates', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def demonstrate_box_matrix_theory():
    """Demonstrate the theory with a simple box matrix example."""
    # Create a simple 2D box matrix
    L = 3.0  # box size
    gamma = 0.4  # shear strain
    
    # Original square box
    H_square = jnp.array([[L, 0.0], 
                          [0.0, L]])
    
    # Sheared box (xy shear)
    H_sheared = jnp.array([[L, gamma * L], 
                           [0.0, L]])
    
    # Show the transformation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Fractional space (always the unit square)
    axes[0].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                       edgecolor='red', facecolor='lightcoral', alpha=0.3))
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_aspect('equal')
    axes[0].set_title('Fractional Coordinates\n(Always Unit Square)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('sₓ')
    axes[0].set_ylabel('sᵧ')
    axes[0].grid(True, alpha=0.3)
    
    # Original real space
    plot_box_and_grid_2d(H_square, ax=axes[1], title='Real Space (Original)', 
                        show_grid=True, box_color='blue')
    
    # Sheared real space  
    plot_box_and_grid_2d(H_sheared, ax=axes[2], title=f'Real Space (Sheared γ={gamma})', 
                        show_grid=True, box_color='green')
    
    plt.tight_layout()
    return fig, H_square, H_sheared


def demonstrate_periodic_boundaries():
    """Demonstrate periodic boundaries under shear."""
    R_boundary = create_boundary_particles()
    
    figures = []
    for gamma in [0.0, 0.5, 1.0]:
        fig, ax = show_periodic_images(gamma, R_boundary, n_images=1)
        figures.append(fig)
    
    return figures


# =============================================================================
# Export Functions for Easy Import
# =============================================================================

__all__ = [
    # Core visualization
    'corners_of_box',
    'grid_lines_2d', 
    'plot_box_and_grid_2d',
    'visualize_coordinate_transformation',
    
    # 3D visualization
    'corners_of_box_3d',
    'box_edges_indices_3d',
    'plot_3d_box',
    'visualize_multi_plane_shear',
    
    # Animation and dynamics
    'create_shear_animation',
    'create_trajectory_animation',
    'unwrap_trajectory',
    
    # Analysis tools
    'plot_stress_evolution',
    'plot_particle_trajectories',
    
    # Shear protocols
    'constant_shear_rate',
    'oscillatory_shear', 
    'step_strain',
    'ramp_strain',
    
    # Periodic boundaries
    'show_periodic_images',
    
    # Comprehensive tools
    'compare_shear_protocols',
    'create_demonstration_grid',
    'create_boundary_particles',
    'plot_static_configuration',
    
    # Tutorial demonstrations
    'demonstrate_coordinate_transformation',
    'demonstrate_box_matrix_theory',
    'demonstrate_periodic_boundaries',
    
    # Viscosity analysis
    'demo_cumulative_viscosity',
    'plot_viscosity_convergence',
]



def rsa_cell_initial_positions(N, phi, a, seed=0, max_trials_per_particle=10000, radii=None):
    """
    Periodic RSA (random sequential addition) with spatial hashing (cell dict).
    Irregular, overlap-free configurations for moderate volume fractions.

    Supports monodisperse (via `a`) and polydisperse/bidisperse (via `radii`).

    Args:
        N: number of particles.
        phi: target volume fraction.
        a: monodisperse particle radius (ignored if `radii` is provided).
        seed: RNG seed.
        max_trials_per_particle: limit to avoid infinite loops at high phi.
        radii: optional array of per-particle radii (length N).

    Returns:
        positions: jnp.ndarray of shape (N, 3) in real-space [0, L)^3.
        L: box length (float64)
    """
    # # Basic input validation
    # sanity_check_inputs(N=N, phi=phi, a=a, radii=radii)

    rng = np.random.default_rng(seed)

    # Predeclare for static analyzers
    radii_sorted = None
    order_desc = None
    if radii is not None:
        radii = np.asarray(radii, dtype=np.float64)
        if radii.shape != (N,):
            raise ValueError(f"radii must have shape ({N},), got {radii.shape}")
        # Store original order so we can return positions consistent with input ordering.
        orig_radii = radii.copy()
        # Sort radii in descending order so larger particles are placed first (reduces rejection rate).
        order_desc = np.argsort(radii)[::-1]
        radii_sorted = radii[order_desc]
        # Compute box size from sum of individual volumes (order independent).
        total_volume = np.sum((4.0/3.0) * np.pi * radii_sorted**3)
        L = float((total_volume / phi) ** (1.0/3.0))
        # Conservative cell size based on the largest contact distance.
        r_max = float(radii_sorted.max())
        min_cell = (r_max + r_max) * 1.0  # small slack
    else:
        a = float(a)
        # Box size from phi (monodisperse)
        particle_volume = (4.0/3.0)*np.pi*a**3
        L = float((N*particle_volume/phi)**(1/3))
        min_cell = 2.0*a * 1.0

    # Cell size >= min_cell so neighbors are within the 27 surrounding cells
    cell = max(min_cell, 1e-12)
    ncell = max(1, int(L / cell))     # integer number of cells along each axis
    cell = L / ncell                  # adjusted cell size

    # Spatial hash: map (i,j,k) -> list of particle indices in that cell
    buckets = defaultdict(list)

    def cell_index(x):
        """x: 3-vector in [0,L). Returns tuple of int indices (i,j,k) in [0, ncell-1]."""
        ijk = np.floor(x / cell).astype(int) % ncell
        return int(ijk[0]), int(ijk[1]), int(ijk[2])

    pts = []
    placed = 0
    trials = 0
    max_trials_total = int(max_trials_per_particle) * int(N)

    while placed < N and trials < max_trials_total:
        trials += 1
        # uniform proposal in the periodic box
        p = rng.random(3) * L
        ci, cj, ck = cell_index(p)

        ok = True
        # check 27 neighboring cells
        for di in (-1, 0, 1):
            if not ok: break
            ni = (ci + di) % ncell
            for dj in (-1, 0, 1):
                if not ok: break
                nj = (cj + dj) % ncell
                for dk in (-1, 0, 1):
                    nk = (ck + dk) % ncell
                    for idx in buckets[(ni, nj, nk)]:
                        q = pts[idx]
                        # minimum-image distance
                        d = p - q
                        d -= L*np.round(d/L)
                        # allowed center distance at contact: r_i + r_j
                        if radii is not None and radii_sorted is not None:
                            # Using sorted radii for placement ordering
                            rij = radii_sorted[placed] + radii_sorted[idx]
                        else:
                            rij = 2.0 * float(a)
                        # small tolerance
                        rij *= 1.0
                        if d.dot(d) < rij**2:
                            ok = False
                            break
                    if not ok: break

        if ok:
            idx_new = len(pts)
            pts.append(p)
            buckets[(ci, cj, ck)].append(idx_new)
            placed += 1
            if placed % max(1, N//10) == 0:
                print(f"Placed {placed}/{N} (trial {trials}, cell {ci},{cj},{ck})", end='\r')

    if placed < N:
        raise RuntimeError(
            f"RSA gave up after {trials} trials (placed {placed}/{N}). "
            "Increase max_trials_per_particle or lower phi."
        )

    pts = np.asarray(pts, dtype=np.float64)
    if radii is not None and order_desc is not None:
        # pts are currently in sorted (descending radius) order. Reorder back to original input order
        pts_original = np.empty_like(pts)
        pts_original[order_desc] = pts  # inverse permutation
        pts = pts_original
        print(f"RSA: Box L={L:.6f}, cells={ncell}^3, trials={trials} (placed big-first)")
    else:
        print(f"RSA: Box L={L:.6f}, cells={ncell}^3, trials={trials}")
    return jnp.asarray(pts), jnp.float64(L)


def demo_cumulative_viscosity(stress_tensor, time, volume, temperature):
    """
    Demonstrate how to compute cumulative viscosity integral.
    
    This function shows how the new cumulative viscosity functionality works,
    allowing you to see how the viscosity converges as a function of integration time.
    
    Args:
        stress_tensor: Stress tensor time series
        time: Time array 
        volume: System volume
        temperature: Temperature
        
    Returns:
        Dictionary with viscosity analysis results including cumulative arrays
    """
    from jax_md import rheo
    
    # Compute Green-Kubo viscosity with cumulative integration
    results = rheo.green_kubo_viscosity(
        stress_tensor=stress_tensor,
        time=time,
        volume=volume,
        temperature=temperature,
        max_modes=10
    )
    
    # Extract results
    final_viscosity = results['viscosity']
    cumulative_fitted = results['cumulative_viscosity_fitted']
    cumulative_raw = results['cumulative_viscosity_raw']
    full_time = results['time']
    fitted_time = results['fitted_time']
    
    print(f"Final converged viscosity: {final_viscosity:.6f}")
    print(f"Cumulative values available for {len(cumulative_fitted)} fitted time points")
    print(f"Cumulative values available for {len(cumulative_raw)} raw time points")
    
    # You can also compute cumulative viscosity directly from autocorrelation
    acf = results['autocorrelation_function'] 
    cumulative_direct = rheo.viscosity_integral_direct(acf, full_time)
    
    print(f"Direct cumulative integral has {len(cumulative_direct)} points")
    print(f"Final value from direct integration: {cumulative_direct[-1]:.6f}")
    
    return {
        'final_viscosity': final_viscosity,
        'cumulative_fitted': cumulative_fitted,
        'cumulative_raw': cumulative_raw,
        'cumulative_direct': cumulative_direct,
        'time_fitted': fitted_time,
        'time_full': full_time,
        'convergence_ratio': cumulative_fitted / final_viscosity  # Shows convergence progress
    }


def plot_viscosity_convergence(cumulative_viscosity, time, title="Viscosity Convergence"):
    """
    Plot how viscosity converges as a function of integration time.
    
    Args:
        cumulative_viscosity: Array of cumulative viscosity values
        time: Corresponding time array
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot absolute cumulative viscosity
    ax1.plot(time, cumulative_viscosity, 'b-', linewidth=2)
    ax1.set_xlabel('Integration Time')
    ax1.set_ylabel('Cumulative Viscosity')
    ax1.set_title(f'{title} - Absolute Values')
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence as percentage of final value
    final_value = cumulative_viscosity[-1]
    convergence_percent = 100 * cumulative_viscosity / final_value
    
    ax2.plot(time, convergence_percent, 'r-', linewidth=2)
    ax2.axhline(y=95, color='k', linestyle='--', alpha=0.5, label='95% convergence')
    ax2.axhline(y=99, color='k', linestyle=':', alpha=0.5, label='99% convergence')
    ax2.set_xlabel('Integration Time')
    ax2.set_ylabel('Convergence (%)')
    ax2.set_title(f'{title} - Convergence Progress')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

