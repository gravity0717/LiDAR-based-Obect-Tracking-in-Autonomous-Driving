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
import matplotlib.pyplot as plt
from utils import create_button, create_label
from event import Event
from association import Association 
from logger import get_logger
import threading    


class Vis:
    def __init__(self):
        self.asso = Association()
        self.logger = get_logger()
        self.lock = threading.Lock()

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
        
        ### buttons
        self.btn_show_axes = create_button("Show axes", (0.1, 1.5), True, None, self.panel)
        
        ### labels
        self.label_voxel_size = create_label("Voxel Size : ", self.panel)
        self.label_distance = create_label("Distance : ", self.panel)
        self.label_ground = create_label("Ground : ", self.panel)
        self.label_frame_number = create_label("Frame: ", self.panel)
        self.label_tracklet_length = create_label("Tracked Cars:", self.panel)
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
    
    def remove_all(self):
        self.geometry.remove()
        
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
        frame = 0 
        try:
            while not event.loop_stop:
                self.dataset = Dataset("data/20220331T160645_ProjectData.mat")
                with self.lock:
                    for t, (pcd, image, left, right, vel) in enumerate(self.dataset):
            
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
                        
                        ### Filter pcd with xy-range 
                        points = np.array(np.asarray(pcd.points)[(np.abs(np.asarray(pcd.points)[:, 0]) < 50) & (np.abs(np.asarray(pcd.points)[:, 1]) < 3)])
                        pcd.points = o3d.utility.Vector3dVector(points)
                        
                        ### Clustering DBSCAN
                        labels = np.array(pcd.cluster_dbscan(eps=event.eps, min_points=10, print_progress=False))
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
                        
                        ### Bounding box와 cluster 정보 수집
                        centroids = []
                        bbox_info = []
                        for i in range(max_label + 1):
                            cluster = pcd.select_by_index(np.where(labels == i)[0])      
                            bbox = cluster.get_axis_aligned_bounding_box()                                         
                            centroid = bbox.get_center()
                            centroids.append(centroid)
                            bbox_info.append((i, bbox))
                            
                        ### app.post_to_main_thread에서 한 번에 업데이트
                        def update_all_geometry_and_text(bbox_info, tracklets):
                            # Bounding box 업데이트
                            for i, bbox in bbox_info:
                                self.geometry.update_geometry(f"bbox_{i}", bbox, event.show_bbox)
                            
                            # 클러스터 ID 업데이트
                            for cluster_id, cluster in tracklets.items():
                                self.geometry.update_text(
                                    str(cluster_id), 
                                    cluster['centroid'], 
                                    f"ID: {cluster_id}, Age: {cluster['age']}"
                                )

                        ### Updates clusters     
                        self.asso.update_clusters(centroids)
                    
                        app.post_to_main_thread(
                            self.window, 
                            lambda: update_all_geometry_and_text(bbox_info, self.asso.tracklets)
                        )
                                                
                        ### Label update
                        self.label_voxel_size.text = f"Voxel size : {event.voxel_size}"
                        self.label_distance.text = f"Eps : {event.eps}"
                        self.label_ground.text = f"Ground : {event.ground}"
                        self.label_frame_number.text = f"Frame : {frame}"
                        self.label_tracklet_length.text= f"tracked Cars : {len(self.asso.tracklets)}"
                        frame += 1 

                        ### Time
                        time.sleep(0.05)
                        
        except Exception as e:
            self.logger.error("Error occurred", exc_info=True)
            event.loop_stop = True
            event.exit_thread = True
            
        event.exit_thread = True
        print("thread done")
                
if __name__ == "__main__":
    vis = Vis()
    vis.run()
