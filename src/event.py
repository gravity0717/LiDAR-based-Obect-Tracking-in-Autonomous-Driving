import open3d.visualization.rendering as rendering

class Event:
    def __init__(self, vis):
        self.vis = vis
        self.widget = vis.widget
        
        ## Window process
        self.loop_stop = False
        self.exit_thread = False
        
        ## data wait
        self.wait = False
        
        ## Visualize
        self.show_ground = False
        self.show_inlier = True
        self.show_outlier = False
        self.test = False
        self.show_bbox = True
        
        ## Values
        self.eps = 0.5
        self.voxel_size = 0.05
        self.ground = 0.27
        
    def on_key(self, keyevent):
        
        if keyevent.type == keyevent.type.DOWN:

            if keyevent.key == 32: # space
                self.wait = not self.wait
                
            if keyevent.key == ord('x'):
                self.show_ground = not self.show_ground
            
            if keyevent.key == ord('w'):
                print(self.widget.scene.camera.get_model_matrix())
    
            if keyevent.key == ord('a'):
                self.widget.scene.camera.set_projection(60, 1.5, 0, 1000, rendering.Camera.FovType.Horizontal)
                self.widget.scene.camera.look_at([2.3, 0, 1.2], [2.2, 0, 1.2], [0, 0, 1])
    
            if keyevent.key == ord('e'):
                self.widget.scene.camera.look_at([0, 0, 0], [-40, 0, 50], [0, 0, 1])
    
            if keyevent.key == ord('t'):
                self.widget.scene.camera.look_at([0, 0, 0], [0, 0, 70], [0, 0, 1])
    
            if keyevent.key == ord('d'):
                self.down = not self.down
            
            if keyevent.key == ord('o'):
                self.show_outlier = not self.show_outlier
            
            if keyevent.key == ord('r'):
                self.test = not self.test            
    
            if keyevent.key == ord('y'):
                self.eps += 0.05
            
            if keyevent.key == ord('h'):
                self.eps -= 0.05
    
            if keyevent.key == ord('u'):
                self.voxel_size += 0.05
            
            if keyevent.key == ord('j'):
                if self.voxel_size - 0.05 <= 1e-14:
                    return
                self.voxel_size -= 0.05
    
            if keyevent.key == ord('i'):
                self.show_inlier = not self.show_inlier
                
            if keyevent.key == ord('c'):
                self.ground -= 0.01
                
            if keyevent.key == ord('v'):
                self.ground += 0.01
            
        
        
        
        
        