import * as vertex from "libshapes/lib/Vertex";

const MAX_DEPTH = 6;

function magnitude(v) {
  return Math.sqrt(Math.pow(v.x, 2) + Math.pow(v.y, 2));
}

function angle(v) {
  return Math.atan2(v.y, v.x);
}

class Item {

  constructor(id, v) { this._id = id;
    this.id = id;
    this.vertex = v;
    this.magnitude = magnitude(this.vertex);
    this.angle = angle(v);
  }

}

class Node {

  constructor(options = {depth: 0, leftBound: null, rightBound: null}) {
    this.depth = options.depth;
    this.leftBound = options.leftBound;
    this.rightBound = options.rightBound;
    this.midpoint = (this.rightBound + this.leftBound)/2;
    this.left = null;
    this.right = null;
    this.bucket = null;
  }

}

export default class VertexTree {

  constructor(options = {leftBound: 0, rightBound: 100}) {
    options.depth = 0;
    this._node = new Node(options);
  }

  insert(id, v) {
    insert(this._node, new Item(id, v));
  }

  find(query) {
    return search(this._node, searchParameters(query)).filter(item => {
      return (vertex.distance(query.origin, item.vertex) < query.radius);
    });
  }

}

function searchParameters(query) {
  const originMagnitude = magnitude(query.origin);
  const originAngle = angle(query.origin);
  //const angleDistance = Math.atan2(query.radius, originMagnitude);
  const angleDistance = Math.asin(query.radius/originMagnitude);
  return {
    magnitudeMin: originMagnitude - query.radius,
    magnitudeMax: originMagnitude + query.radius,
    angleMin: originAngle - angleDistance,
    angleMax: originAngle + angleDistance
  };
}

function search(node, parameters) {
  if (node.bucket !== null) {
    return node.bucket.filter(item => {
      const angle = item.angle;
      return (angle > parameters.angleMin && angle < parameters.angleMax);
    });
  } else {
    let leftItems = [], rightItems = [];
    if (node.left !== null && parameters.magnitudeMin <= node.midpoint) {
      leftItems = search(node.left, parameters);
    }
    if (node.right !== null && parameters.magnitudeMax > node.midpoint) {
      leftItems = leftItems.concat(search(node.right, parameters));
    }
    return leftItems;
  }
}

function insert(node, item) {
  if (node.depth < MAX_DEPTH) {
    if (item.magnitude < node.midpoint) {
      if (node.left === null) {
        node.left = new Node({
          depth: node.depth + 1,
          leftBound: node.leftBound,
          rightBound: node.midpoint
        });
      }
      insert(node.left, item);
    } else {
      if (node.right === null) {
        node.right = new Node({
          depth: node.depth + 1,
          leftBound: node.midpoint,
          rightBound: node.rightBound
        });
      }
      insert(node.right, item);
    }
  } else {
    if (node.bucket === null) node.bucket = [];
    node.bucket.push(item)
    // Sort is irrelevant at the moment, but could be used for a binary search
    // later.
    //node.bucket.sort((a, b) => { return (a.angle < b.angle) ? -1 : 1; });
  }
}
