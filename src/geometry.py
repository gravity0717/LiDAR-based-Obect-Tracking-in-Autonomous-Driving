import open3d as o3d
import open3d.visualization.rendering as rendering
from src.material import Material
import numpy as np
import time

class Geometry:
    def __init__(self, widget3d):
        self.widget3d = widget3d
        
    def draw_car(self, name, pose, length, width, height):
        
        car = o3d.geometry.LineSet()
        car.points = o3d.utility.Vector3dVector([
          [length / 2, width /2, height], [length / 2, -width / 2, height],
          [-length / 2, -width / 2, height], [-length / 2, width / 2, height],
          [length / 2, width /2, 0], [length / 2, -width / 2, 0],
          [-length / 2, -width / 2, 0], [-length / 2, width / 2, 0]  
        ])
        car.lines = o3d.utility.Vector2iVector([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        car.paint_uniform_color([1.0, 0.0, 0.0])
        self.widget3d.scene.add_geometry(name, car, Material.default)
        self.widget3d.scene.set_geometry_transform(name, pose)
    
    def draw_spline(self, name, a, b, c, color=[1.0, 0.0, 0.0], min=-30, max=30):
        
        self.remove_geometry(name)
        
        spline = o3d.geometry.LineSet()
        spline.points = o3d.utility.Vector3dVector([
            [i, a*i*i + b * i + c, 0] for i in range(min, max, 1)
        ])
        spline.lines = o3d.utility.Vector2iVector([
            [i, i+1] for i in range(0, max-min-1)
        ])
        spline.paint_uniform_color(color)
        self.widget3d.scene.add_geometry(name, spline, Material.default)
    
    def draw_primitives(self):

        self.draw_car("my_car", np.eye(4), 3.6, 1.8, 1.7)
        
        self.widget3d.scene.show_axes(True)
        self.widget3d.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XY)
        bounds = self.widget3d.scene.bounding_box 
        self.widget3d.scene.camera.look_at([0, 0, 0], [-40, 0, 50], [0, 0, 1])
            
    def update_geometry(self, name, geometry, show=True, mat=Material.default):
        self.remove_geometry(name)
        if show:
            self.widget3d.scene.add_geometry(name, geometry, mat)
    
    def remove_geometry(self, name):
        if self.widget3d.scene.has_geometry(name):
            self.widget3d.scene.remove_geometry(name)
            print(f"Removed geometry: {name}")  # 로그 추가
        else:
            print(f"Geometry {name} does not exist")  # 존재하지 않을 경우 로그
    
    def update_text(self, name, coord, text):
        
        coord = np.concatenate((coord, np.array([0])))
        self.remove_text(name)  
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.1)    
        text_mesh.scale(0.2, coord)
        
        # Check numpy is not empty 
        if np.any(coord):
            # -90 degrees rotation around the z-axis
            R = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 1]], dtype=np.float32)
            R_tensor = o3d.core.Tensor(R, dtype=o3d.core.Dtype.Float32)
            coord= o3d.core.Tensor(coord, dtype=o3d.core.Dtype.Float32)
            text_mesh.rotate(R_tensor, coord)     
        self.widget3d.scene.add_geometry(name, text_mesh, Material.default)
        
    def remove_text(self, name):
        if self.widget3d.scene.has_geometry(name):
            self.remove_geometry(name)
            print(f"Removed text: {name}")  # 로그 추가
        else:
            print(f"Text {name} does not exist")  # 존재하지 않을 경우 로그