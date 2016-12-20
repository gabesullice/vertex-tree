"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.VertexTree = exports.Item = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _Vertex = require("libshapes/lib/Vertex");

var vertex = _interopRequireWildcard(_Vertex);

var _Edge = require("libshapes/lib/Edge");

var edges = _interopRequireWildcard(_Edge);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var MAX_DEPTH = 6;

function magnitude(v) {
  return Math.sqrt(Math.pow(v.x, 2) + Math.pow(v.y, 2));
}

function angle(v) {
  return Math.atan2(v.y, v.x);
}

var Item = exports.Item = function () {
  function Item(v) {
    var _ref = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {},
        _ref$edges = _ref.edges,
        edges = _ref$edges === undefined ? [] : _ref$edges;

    _classCallCheck(this, Item);

    this.vertex = v;
    this.magnitude = magnitude(this.vertex);
    this.angle = angle(v);
    this.edges = edges;
  }

  _createClass(Item, [{
    key: "addEdge",
    value: function addEdge(insert) {
      if (this.edges.find(function (edge) {
        return edges.same(insert, edge);
      }) === undefined) {
        this.edges.push(insert);
      };
    }
  }, {
    key: "removeEdge",
    value: function removeEdge(remove) {
      this.edges = this.edges.filter(function (edge) {
        return !edges.same(remove, edge);
      });
    }
  }]);

  return Item;
}();

var Node = function Node() {
  var options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : { depth: 0, leftBound: null, rightBound: null };

  _classCallCheck(this, Node);

  this.depth = options.depth;
  this.leftBound = options.leftBound;
  this.rightBound = options.rightBound;
  this.midpoint = (this.rightBound + this.leftBound) / 2;
  this.left = null;
  this.right = null;
  this.bucket = null;
};

var VertexTree = exports.VertexTree = function () {
  function VertexTree() {
    var options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : { leftBound: 0, rightBound: 100 };

    _classCallCheck(this, VertexTree);

    options.depth = 0;
    this._node = new Node(options);
  }

  _createClass(VertexTree, [{
    key: "insert",
    value: function insert(item, options) {
      if (item instanceof Item) {
        _insert2(this._node, item);
        return true;
      } else if (item instanceof vertex.Vertex) {
        _insert2(this._node, new Item(item, options));
        return true;
      }
      return false;
    }
  }, {
    key: "remove",
    value: function remove(item) {
      return _remove(this._node, item);
    }
  }, {
    key: "at",
    value: function at(v) {
      return _at(this._node, v);
    }
  }, {
    key: "find",
    value: function find(query) {
      return search(this._node, searchParameters(query)).filter(function (item) {
        return vertex.distance(query.origin, item.vertex) < query.radius;
      });
    }
  }, {
    key: "nearest",
    value: function nearest(to) {
      return _nearest(this._node, to);
    }
  }, {
    key: "newItem",
    value: function newItem(v, options) {
      return new Item(v, options);
    }
  }, {
    key: "insertEdge",
    value: function insertEdge(insert) {
      var _this = this;

      insert.vertices().forEach(function (v) {
        var item = _this.at(v);
        if (item) {
          item.addEdge(insert);
        } else {
          _this.insert(v, { edges: [insert] });
        }
      });
    }
  }]);

  return VertexTree;
}();

function searchParameters(query) {
  var originMagnitude = magnitude(query.origin);
  var originAngle = angle(query.origin);
  var angleDistance = void 0;
  if (originMagnitude == 0) {
    angleDistance = Math.PI;
  } else {
    angleDistance = Math.asin(query.radius / originMagnitude);
  }
  return {
    magnitudeMin: originMagnitude - query.radius,
    magnitudeMax: originMagnitude + query.radius,
    angleMin: originAngle - angleDistance,
    angleMax: originAngle + angleDistance
  };
}

function _at(node, term) {
  var bottom = function bottom(node) {
    return node.bucket.find(function (item) {
      return vertex.same(item.vertex, term);
    });
  };

  var into = intoExact(term);

  var out = function out(result) {
    return result;
  };

  return treeTraverser(searchBase, bottom, into, out)(node);
}

function intoExact(term) {
  var termMagnitude = magnitude(term);
  return function (node, recurse) {
    if (node.left !== null && termMagnitude < node.midpoint) {
      return recurse(node.left);
    } else if (node.right !== null) {
      return recurse(node.right);
    }
    return null;
  };
}

function _nearest(node, term) {
  var parameters = {
    magnitudeMin: 0,
    magnitudeMax: node.rightBound,
    angleMin: 0,
    angleMax: Math.PI / 2
  };

  var minimumDistance = void 0;
  var insertBottom = function insertBottom(node) {
    return node.bucket.filter(function (item) {
      var angle = item.angle;
      return angle > parameters.angleMin && angle < parameters.angleMax;
    }).reduce(function (found, item) {
      var distance = vertex.distance(term, item.vertex);
      if (minimumDistance === undefined || distance < minimumDistance) {
        item._distance = distance;
        minimumDistance = distance;
        parameters = searchParameters({ origin: term, radius: distance });
        parameters = searchParameters({ origin: term, radius: distance });
        found = item;
      }
      return found;
    }, []);
  };

  return [treeTraverser(searchBase, insertBottom, searchInto(parameters, this), function (left, right) {
    if (Array.isArray(left) && Array.isArray(right)) {
      return [];
    } else if (Array.isArray(left)) {
      return right;
    } else if (Array.isArray(right)) {
      return left;
    } else {
      return left._distance < right._distance ? left : right;
    }
  })(node)];
}

function search(node, parameters) {
  return treeTraverser(searchBase, searchBottom(parameters), searchInto(parameters), searchOutof)(node);
}

function treeTraverser(base, bottom, into, outof) {
  var self = function self(node) {
    if (base(node)) {
      return bottom(node);
    } else {
      return outof(into(node, self));
    }
  };
  return self;
}

function searchBase(node) {
  return node.bucket !== null;
}

function searchBottom(parameters) {
  return function (node) {
    return node.bucket.filter(function (item) {
      var angle = item.angle;
      return angle > parameters.angleMin && angle < parameters.angleMax;
    });
  };
}

function searchInto(parameters) {
  return function (node, recurse) {
    var leftItems = [],
        rightItems = [];
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

function _insert2(node, item) {
  var base = function base(node) {
    return node.depth >= MAX_DEPTH;
  };

  var bottom = function bottom(node) {
    if (node.bucket === null) node.bucket = [];
    node.bucket.push(item);
  };

  var into = function into(node) {
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

  var outof = function outof() {};

  var _insert = treeTraverser(base, bottom, into, outof);

  _insert(node);
}

function _remove(node, term) {
  var bottom = function bottom(node) {
    var before = node.bucket.length;
    node.bucket = node.bucket.filter(function (item) {
      return !vertex.same(item.vertex, term);
    });
    return before > node.bucket.length;
  };

  var outof = function outof(result) {
    return result ? true : false;
  };

  var _insert = treeTraverser(searchBase, bottom, intoExact(term), outof);

  _insert(node);
}