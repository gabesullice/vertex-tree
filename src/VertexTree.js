import * as vertex from "libshapes/lib/Vertex";
import * as edges from "libshapes/lib/Edge";

const MAX_DEPTH = 6;

function magnitude(v) {
  return Math.sqrt(Math.pow(v.x, 2) + Math.pow(v.y, 2));
}

function angle(v) {
  return Math.atan2(v.y, v.x);
}

export class Item {

  constructor(id, v, {edges = 0} = {}) {
    this._id = id;
    this.id = id;
    this.vertex = v;
    this.magnitude = magnitude(this.vertex);
    this.angle = angle(v);
    this.edges = edges;
  }

  addEdge(insert) {
    if (this.edges.find(edge => {
      return edges.same(insert, edge);
    }) === undefined) {
      this.edges.push(insert);
    };
  }

  removeEdge(remove) {
    this.edges = this.edges.filter(edge => {
      return !edges.same(remove, edge);
    })
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

export class VertexTree {

  constructor(options = {leftBound: 0, rightBound: 100}) {
    options.depth = 0;
    this._node = new Node(options);
  }

  insert(id, v) {
    if (id instanceof Item) {
      insert(this._node, id);
    } else {
      insert(this._node, new Item(id, v));
    }
  }

  find(query) {
    return search(this._node, searchParameters(query)).filter(item => {
      return (vertex.distance(query.origin, item.vertex) < query.radius);
    });
  }

  nearest(to) {
    return nearest(this._node, to)
  }

  newItem(id, v, options) {
    return new Item(id, v, options);
  }

  addEdge(insert) {
    insert.vertices().forEach(v => {
      const item = this._itemAt(v);
      if (item) {
        item.addEdge(edge);
      }
    });
  }

  _itemAt(v) {

  }

}

function searchParameters(query) {
  const originMagnitude = magnitude(query.origin);
  const originAngle = angle(query.origin);
  const angleDistance = Math.asin(query.radius/originMagnitude);
  return {
    magnitudeMin: originMagnitude - query.radius,
    magnitudeMax: originMagnitude + query.radius,
    angleMin: originAngle - angleDistance,
    angleMax: originAngle + angleDistance
  };
}

function nearest(node, term) {
  let parameters = {
    magnitudeMin: 0,
    magnitudeMax: node.rightBound,
    angleMin: 0,
    angleMax: Math.PI/2,
  };

  let minimumDistance;
  const insertBottom = function (node) {
    return node.bucket.filter(item => {
      const angle = item.angle;
      return (angle > parameters.angleMin && angle < parameters.angleMax);
    }).reduce((found, item) => {
      const distance = vertex.distance(term, item.vertex);
      if (minimumDistance === undefined || distance < minimumDistance) {
        item._distance = distance;
        minimumDistance = distance;
        parameters = searchParameters({origin: term, radius: distance});
        parameters = searchParameters({origin: term, radius: distance});
        found = item;
      } 
      return found;
    }, []);
  }

  return [treeTraverser(
    searchBase,
    //searchBottom(parameters),
    insertBottom,
    searchInto(parameters, this),
    function (left, right) {
      if (Array.isArray(left) && Array.isArray(right)) {
        return [];
      } else if (Array.isArray(left)) {
        return right;
      } else if (Array.isArray(right)) {
        return left;
      } else {
        return (left._distance < right._distance) ? left : right;
      }
    },
  )(node)];
}

function search(node, parameters) {
  return treeTraverser(
    searchBase,
    searchBottom(parameters),
    searchInto(parameters),
    searchOutof
  )(node);
}

function treeTraverser(base, bottom, into, outof) {
  const self = function(node) {
    if (base(node)) {
      return bottom(node);
    } else {
      return outof(...into(node, self));
    }
  }
  return self;
}

function searchBase(node) {
  return node.bucket !== null;
}

function searchBottom(parameters) {
  return function (node) {
    return node.bucket.filter(item => {
      const angle = item.angle;
      return (angle > parameters.angleMin && angle < parameters.angleMax);
    });
  };
}

function searchInto(parameters) {
  return function (node, recurse) {
    let leftItems = [], rightItems = [];
    if (node.left !== null && parameters.magnitudeMin <= node.midpoint) {
      leftItems = recurse(node.left, parameters);
    }
    if (node.right !== null && parameters.magnitudeMax > node.midpoint) {
      rightItems = recurse(node.right, parameters);
    }
    return [leftItems, rightItems];
  };
}

function searchOutof(left, right) {
  return left.concat(right);
};

function insert(node, item) {
  const base = function (node) {
    return node.depth >= MAX_DEPTH;
  };

  const bottom = function (node) {
    if (node.bucket === null) node.bucket = [];
    node.bucket.push(item)
  };

  const into = function (node) {
    if (item.magnitude < node.midpoint) {
      if (node.left === null) {
        node.left = new Node({
          depth: node.depth + 1,
          leftBound: node.leftBound,
          rightBound: node.midpoint
        });
      }
      _insert(node.left, item);
    } else {
      if (node.right === null) {
        node.right = new Node({
          depth: node.depth + 1,
          leftBound: node.midpoint,
          rightBound: node.rightBound
        });
      }
      _insert(node.right, item);
    }
    return [];
  };

  const outof = function () {};

  const _insert = treeTraverser(base, bottom, into, outof);

  _insert(node);
}
