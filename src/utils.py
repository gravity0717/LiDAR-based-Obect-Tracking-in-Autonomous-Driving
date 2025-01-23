import open3d.visualization.gui as gui
import open3d as o3d
import numpy as np

def create_button(name, paddings, toggleable=False, fn=None, parent=None):
    btn = gui.Button(name)
    btn.horizontal_padding_em = paddings[0]
    btn.vertical_padding_em = paddings[1]
    btn.toggleable = toggleable
    if fn is not None:
        btn.set_on_clicked(fn)
    if parent is not None:
        parent.add_child(btn)
    
    return btn

def create_label(label, parent=None):
    label = gui.Label(label)
    if parent is not None:
        parent.add_child(label)
    
    return label
