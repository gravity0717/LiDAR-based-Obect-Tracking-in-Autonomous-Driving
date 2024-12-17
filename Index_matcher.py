import numpy as np
import open3d as o3d



class IndexMatcher:
    def __init__(self):
        self.table_index = None 
        self.tracklet_index = None
        self.untracked_index = None 
        self.matcher = []
    
    def set_tracklet_index(self, tracklet_index):
        # tracklet_index: tracklet의 index
        self.tracklet_index = tracklet_index  
          
    def set_match(self):    
        self.matcher = [(i,j) for i,j in enumerate(self.tracklet_index)]
    
    def convert_table2tracklet(self, table_index):
        for i, j in self.matcher:
            if i == table_index:
                return j
        return None
    
        

    
        
        