from itertools import chain
from builtins import map
import random
import time
import cv2
import math

class Node(object):

    def __init__(self, start, goal, cost):
        self.distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        self.start = start  # Node position
        self.goal = goal    # Goal position
        self.id = id(self)  # Unique identifier for class instance
        self.children = []
        self.parent = None  # Used to trace lineage from goal back to start
        self.done = False   # Only true for goal node

    def __iter__(self):
        for v in chain(*map(iter, self.children)):
            yield v
        yield self

    def add_child(self, obj):
        self.children.append(obj)


def expand_tree(node):

    pathReversed = []
    pathReversed.append(node.start)
    while node.parent is not None:
        pathReversed.append(node.parent.start)
        node = node.parent

    return pathReversed


def generate_node(workspace, parent, sample, stepSize, fin=False):
    (h, w) = workspace.shape
    angle = math.atan2(parent.start[1] - sample[1], parent.start[0] - sample[0])
    newX = parent.start[0] - stepSize * random.random() * math.cos(angle)
    newY = parent.start[1] - stepSize * random.random() * math.sin(angle)
    if fin:
        newX = sample[0]
        newY = sample[1]

    if workspace[int(newY * h), int(newX * w)] == 255:
        update = Node((newX, newY), parent.goal, math.hypot(newX - parent.goal[0], newY - parent.goal[1]))
        update.parent = parent
        return update
    return None


def generate_path(canvas, startPixel, endPixel, stepSize, plot=False):

    # Convert start and end pixels to node locations
    (w, h) = canvas.shape
    startNode = (float(startPixel[1]) / h, float(startPixel[0]) / w)
    endNode = (float(endPixel[1]) / h, float(endPixel[0]) / w)

    # Store euclidean distance heuristic in root node
    newTree = Node(startNode, endNode, 0)
    newTree.root = startNode
    finalPoint = False

    # Set timeout value
    timeout = time.time() + 1

    while True:
        if not finalPoint:
            # Generate valid sample while candidate point not in collision with world
            samplePoint = (random.random(), random.random())

        # Select closest node to sample point by iterating through nodes
        closestNode = 1
        for node in iter(newTree):

            # Check for win condition
            if node.distance == 0:
                path = expand_tree(node)
                return path

            sampleDistance = math.hypot(samplePoint[0] - node.start[0], samplePoint[1] - node.start[1])
            if sampleDistance <= closestNode:
                closestNode = sampleDistance

        # And then update node with sample point at position
        for node in iter(newTree):
            sampleDistance = math.hypot(samplePoint[0] - node.start[0], samplePoint[1] - node.start[1])
            if sampleDistance == closestNode:   # add child
                newNode = generate_node(canvas, node, samplePoint, stepSize, finalPoint)

                if newNode is not None:
                    node.add_child(newNode)
                    newNode.root = node.root

                if node.distance <= stepSize:
                    finalPoint = True
                    samplePoint = node.goal

        # Break if time exceeds timeout
        if time.time() > timeout:
            return None
