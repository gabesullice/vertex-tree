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

  constructor(v, {edges = []} = {}) {
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
    const before = this.edges.length;
    this.edges = this.edges.filter(edge => {
      return !edges.same(remove, edge);
    })
    return before > this.edges.length;
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

  insert(item, options) {
    if (item instanceof Item) {
      insert(this._node, item);
      return true;
    } else if (item instanceof vertex.Vertex) {
      insert(this._node, new Item(item, options));
      return true;
    }
    return false;
  }

  remove(item) {
    return remove(this._node, item);
  }

  at(v) {
    return at(this._node, v);
  }

  find(query) {
    return search(this._node, searchParameters(query)).filter(item => {
      return (vertex.distance(query.origin, item.vertex) < query.radius);
    });
  }

  nearest(to) {
    return nearest(this._node, to)
  }

  newItem(v, options) {
    return new Item(v, options);
  }

  insertEdge(insert) {
    insert.vertices().forEach(v => {
      const item = this.at(v);
      if (item) {
        item.addEdge(insert);
      } else {
        this.insert(v, {edges: [insert]});
      }
    });
  }

  removeEdge(remove) {
    return remove.vertices().reduce((removed, v) => {
      let didRemove = false;
      const item = this.at(v);
      if (item) didRemove = item.removeEdge(remove);
      return (removed || didRemove) ? true : false;
    }, false);
  }

}

function searchParameters(query) {
  const originMagnitude = magnitude(query.origin);
  const originAngle = angle(query.origin);
  let angleDistance;
  if (originMagnitude == 0) {
    angleDistance = Math.PI;
  } else {
    angleDistance = Math.asin(query.radius/originMagnitude);
  }
  return {
    magnitudeMin: originMagnitude - query.radius,
    magnitudeMax: originMagnitude + query.radius,
    angleMin: originAngle - angleDistance,
    angleMax: originAngle + angleDistance
  };
}

function at(node, term) {
  const bottom = function (node) {
    return node.bucket.find(item => {
      return vertex.same(item.vertex, term);
    });
  }

  const into = intoExact(term);

  const out = function (result) {
    return result;
  }

  return treeTraverser(
    searchBase,
    bottom,
    into,
    out,
  )(node);
}

function intoExact(term) {
  const termMagnitude = magnitude(term);
  return function (node, recurse) {
    if (node.left !== null && termMagnitude < node.midpoint) {
      return recurse(node.left);
    } else if (node.right !== null) {
      return recurse(node.right);
    }
    return undefined;
  }
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
      return outof(into(node, self));
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

function searchOutof(tuple) {
  return tuple[0].concat(tuple[1]);
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

function remove(node, term) {
  const bottom = function (node) {
    const before = node.bucket.length;
    node.bucket = node.bucket.filter(item => {
      return !vertex.same(item.vertex, term);
    })
    return before > node.bucket.length; 
  };

  const outof = function (result) {
    return (result) ? true : false;
  };

  const _insert = treeTraverser(
    searchBase,
    bottom,
    intoExact(term),
    outof
  );

  _insert(node);
}
