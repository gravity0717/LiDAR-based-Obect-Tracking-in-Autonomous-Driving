from scipy import io
import numpy as np
import open3d as o3d

class Dataset:
    def __init__(self, mat_path):
        self.datas = io.loadmat(mat_path)
        self.cur = 0
    
    def __len__(self):
        return len(self.datas["Ibeo_X_stack"][0])
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur >= len(self):
            raise StopIteration

        x = self.datas["Ibeo_X_stack"][0][self.cur]
        y = self.datas["Ibeo_Y_stack"][0][self.cur]
        z = self.datas["Ibeo_Z_stack"][0][self.cur]
        
        left_line_poly = self.datas["poly_vision_left_stack"][0][self.cur][0]
        right_line_poly = self.datas["poly_vision_right_stack"][0][self.cur][0]

        velocity = self.datas["Vx_stack"][0][self.cur]
        
        image = self.datas["dash_cam_stack"][0][self.cur][0][0][0]
        image = np.flipud(image)
        image = np.fliplr(image)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        xyz = np.vstack([x, y, z])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.T)
        
        o3d_image = o3d.geometry.Image(image)
        
        self.cur += 1
        return pcd, o3d_image, left_line_poly, right_line_poly, velocity
    
    def wait(self):
        self.cur -= 1