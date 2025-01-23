import open3d.visualization.rendering as rendering
import open3d as o3d

class Material:
    
    default = rendering.Material() if o3d.__version__ == "0.13.0" else rendering.MaterialRecord()
    default.shader = 'defaultUnlit'
    default.sRGB_color = True

    