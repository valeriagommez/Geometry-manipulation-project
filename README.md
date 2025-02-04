# Mesh manipulation project
## Overview
This project involves implementing a **Half-Edge Data Structure** for mesh editing, allowing for various edge and face operations such as flipping, splitting, extruding, and more. The final goal is to utilize these operations to design a 3D object and export it as a mesh file.

## Features
- Construct a **Half-Edge Data Structure** from a given mesh.
- Implement key **edge operations**:
  - Flip edge
  - Split edge
  - Erase edge (merge adjacent faces)
- Implement key **face operations**:
  - Inset face
  - Extrude face
  - Bevel face
- Triangulate convex polygons.
- Interactive visualization and manipulation of meshes.
- Save the final modified mesh to a file.

## Controls
| Key | Action |
|------|------------------------------------------------|
| **F** | Flip selected edge (if an edge is selected) |
| **S** | Split selected edge (if an edge is selected) |
| **E** | Hold for Face Extrude Mode (if a face is selected) |
| **B** | Hold for Face Bevel Mode (if a face is selected) |
| **I** | Hold for Face Inset Mode (if a face is selected) |
| **X** | Set view to X-axis (side view) |
| **Y** | Set view to Y-axis (top view) |
| **Z** | Set view to Z-axis (front view) |
| **Left-Click** | Select an element on the mesh |
| **Click + Drag** | Rotate the camera or perform face operations |
| **Scroll** | Zoom in/out |
| **W** | Save the mesh to a file |
| **ESC** | Close the application |


## Author
- **Valeria Gomez**  
