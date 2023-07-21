import math
from enum import Enum


class FingerID(Enum):
    F1 = "60"
    F2 = "120"
    F3 = "300"


class Point3D():
    x = None
    y = None
    z = None
    
    @property
    def get_pos(self):
        return [self.x, self.y, self.z]


class Finger():
    def __init__(self, ID, hz, timeout=250):
        self.ID = ID
        self.pos = Point3D()
        self.old_pos = Point3D()
        self.g = Point3D()
        self.colour = None
        self.dt = 1/hz
        self.timeout = timeout
        self.reset_value = timeout
        
    def set_pos(self, x, y, z):
        # Store old position 
        self.old_pos.x = self.pos.x
        self.old_pos.y = self.pos.y
        self.old_pos.z = self.pos.z
        
        # Store new position 
        self.pos.x = x
        self.pos.y = y
        self.pos.z = z
        
    def set_goal(self, x, y, z):
        self.g.x = x
        self.g.y = y
        self.g.z = z
        
    def set_colour(self, r, g, b):
        self.colour = (r, g, b)
        
    def dec_timeout(self):
        self.timeout -= 1
        
    def reset_timeout(self):
        self.timeout = self.reset_value
    
    @property
    def dg(self):
        if self.pos.x is None or self.g.x is None:
            return None
        else:
            return math.dist(self.pos.get_pos, self.g.get_pos)
             
    @property
    def v(self):
        if self.pos.x is None or self.old_pos.x is None:
            return None
        else:
            vx = (self.pos.x - self.old_pos.x) / self.dt
            vy = (self.pos.y - self.old_pos.y) / self.dt
            vz = (self.pos.z - self.old_pos.z) / self.dt
            return math.sqrt(vx**2 + vy**2 + vz**2)
       
    @property 
    def intensity(self):
        R, G, B = self.colour
        intensity = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return intensity
    
    
    def dist2D(self, p):
        if self.pos.x is None or self.g.x is None:
            return None
        else:
            return math.dist(self.pos.get_pos[0:2], p[0:2])