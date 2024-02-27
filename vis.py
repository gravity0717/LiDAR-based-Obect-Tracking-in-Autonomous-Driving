import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
from scipy import io
import time
from geometry import Geometry
from dataset import Dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import create_button, create_label
from event import Event

class Vis:
    def __init__(self):
        pass
    
    def run(self):
        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window(
            "Autonomous Driving Term Project", 1280, 720
        )
        
        ### Panel
        vspacing = 4
        margins = gui.Margins(10, 10, 10, 10)
        self.panel = gui.Vert(vspacing, margins)
        self.input_color_image = gui.ImageWidget()
        self.panel.add_child(self.input_color_image)
        
        # buttons
        self.btn_show_axes = create_button("Show axes", (0.1, 1.5), True, None, self.panel)
        
        # labels
        self.label_voxel_size = create_label("Voxel Size : ", self.panel)
        self.label_distance = create_label("Distance : ", self.panel)
        self.label_ground = create_label("Ground : ", self.panel)

        self.window.add_child(self.panel)
        
        ### Widget
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([0, 0, 0, 0])
        self.geometry = Geometry(self.widget)
        self.geometry.draw_primitives()
        self.window.add_child(self.widget)
        
        ### Window event
        self.event = Event(self)
        self.window.set_on_layout(self.layout)
        self.window.set_on_close(self.close)
        self.window.set_on_key(self.event.on_key)

        ### Thread context
        time.sleep(1)
        threading.Thread(target=self.context, daemon=True).start()
        app.run()
    
    def layout(self, ctx):
        rect = self.window.content_rect
        self.panel.frame = gui.Rect(rect.x, rect.y, rect.width // 4, rect.height)        
        x = self.panel.frame.get_right()
        self.widget.frame = gui.Rect(x, rect.y, rect.get_right() - x, rect.height)
    
    def close(self):
        self.event.loop_stop = True
        
        if not self.event.exit_thread:
            time.sleep(0.1)
        self.window.close()
        return True
    
    def context(self):
        
        # For concise
        app = gui.Application.instance        
        event = self.event

        while not event.loop_stop:
            self.dataset = Dataset("data/20220331T160645_ProjectData.mat")
            
            for pcd, image, left, right, vel in self.dataset:
                if event.loop_stop:
                    break

                if event.wait:
                    self.dataset.wait()

                ### Ground filtering
                if not event.show_ground:
                    points = np.asarray(pcd.points)
                    points = points[points[:, 2] > event.ground]
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color([0.1, 0.2, 0.3])
                
                ### Voxel down sampling
                pcd = pcd.voxel_down_sample(event.voxel_size)
                
                ### Clustering DBSCAN
                labels = np.array(pcd.cluster_dbscan(eps=event.eps, min_points=5, print_progress=False))
                max_label = labels.max()
                colors = plt.get_cmap("tab20")(labels/(max_label if max_label > 0 else 1))
                colors[labels < 0] = 0 if event.test else 1
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
                ### Model Point cloud update
                app.post_to_main_thread(
                    self.window, lambda : self.geometry.update_geometry("pcd", pcd, event.show_inlier)
                )
                
                ### Image update
                app.post_to_main_thread(
                    self.window, lambda : self.input_color_image.update_image(image)
                )
                
                ### Lines
                app.post_to_main_thread(
                    self.window, lambda : self.geometry.draw_spline("left", left[0], left[1], left[2], color=[0, 1, 0])
                )
                app.post_to_main_thread(
                    self.window, lambda : self.geometry.draw_spline("right", right[0], right[1], right[2], color=[0, 0, 1])
                )
                
                ### Number 
                ids = range(4)
                for id in ids:
                    # Get pcd coordinate
                    pcd_coord = np.asarray(pcd.points)[labels == id]
                    centroid = np.mean(pcd_coord, axis=0)
                    app.post_to_main_thread(
                        self.window, lambda : self.geometry.add_text(centroid, f"ID : {id}")
                    )
                
                ### Label update
                self.label_voxel_size.text = f"Voxel size : {event.voxel_size}"
                self.label_distance.text = f"Eps : {event.eps}"
                self.label_ground.text = f"Ground : {event.ground}"
                
                ### Time
                time.sleep(0.05)
            
            
        event.exit_thread = True
        print("thread done")
                
if __name__ == "__main__":
    vis = Vis()
    vis.run()