"""
DNN planner
"""

import numpy as np
from .. import Action

class DNNPlanner():
    LENGTH = 0.58
    LOOKAHEAD_DIST = 3
    MAX_SPEED = 5
    MIN_SPEED = 0
    def __init__(self):
        self.TargetPoint = np.array([0., 0.])


    def plan(self, obs, lidarModel, _=1):
        # angle = np.arctan2(-self.TargetPoint[0], self.TargetPoint[1])
        w, pos, rot = lidarModel[0], lidarModel[1], lidarModel[2]
        targetPoint = np.array([w / 2, self.LOOKAHEAD_DIST]) # this is relative to track's coordinate frame
        carRotX = np.array([np.cos(rot), np.sin(rot)])
        carRotY = np.array([np.cos(rot + np.pi/2), np.sin(rot + np.pi/2)])
        carRot = np.column_stack([carRotX, carRotY])
        self.TargetPoint = carRot.transpose().dot(targetPoint - np.array([pos, 0]))
        targetPoint = self.TargetPoint + np.array([0, self.LENGTH]) # relative to back of car
        angle = np.arctan2(-targetPoint[0], targetPoint[1])
        distance = np.linalg.norm(targetPoint)
        steeringAngle = np.arctan(2*self.LENGTH*np.sin(angle)/distance)

        speedInterpolation = 2*abs(steeringAngle)/np.pi
        speed = self.MAX_SPEED + speedInterpolation*(self.MIN_SPEED - self.MAX_SPEED)

        return Action(steer=steeringAngle, speed=speed)
