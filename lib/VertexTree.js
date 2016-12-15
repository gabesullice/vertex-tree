"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _Vertex = require("libshapes/lib/Vertex");

var vertex = _interopRequireWildcard(_Vertex);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var MAX_DEPTH = 6;

function magnitude(v) {
  return Math.sqrt(Math.pow(v.x, 2) + Math.pow(v.y, 2));
}

function angle(v) {
  return Math.atan2(v.y, v.x);
}

var Item = function Item(id, v) {
  _classCallCheck(this, Item);

  this._id = id;
  this.id = id;
  this.vertex = v;
  this.magnitude = magnitude(this.vertex);
  this.angle = angle(v);
};

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

var VertexTree = function () {
  function VertexTree() {
    var options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : { leftBound: 0, rightBound: 100 };

    _classCallCheck(this, VertexTree);

    options.depth = 0;
    this._node = new Node(options);
  }

  _createClass(VertexTree, [{
    key: "insert",
    value: function insert(id, v) {
      _insert(this._node, new Item(id, v));
    }
  }, {
    key: "find",
    value: function find(query) {
      return search(this._node, searchParameters(query)).filter(function (item) {
        return vertex.distance(query.origin, item.vertex) < query.radius;
      });
    }
  }]);

  return VertexTree;
}();

exports.default = VertexTree;


function searchParameters(query) {
  var originMagnitude = magnitude(query.origin);
  var originAngle = angle(query.origin);
  //const angleDistance = Math.atan2(query.radius, originMagnitude);
  var angleDistance = Math.asin(query.radius / originMagnitude);
  return {
    magnitudeMin: originMagnitude - query.radius,
    magnitudeMax: originMagnitude + query.radius,
    angleMin: originAngle - angleDistance,
    angleMax: originAngle + angleDistance
  };
}

function search(node, parameters) {
  if (node.bucket !== null) {
    return node.bucket.filter(function (item) {
      var angle = item.angle;
      return angle > parameters.angleMin && angle < parameters.angleMax;
    });
  } else {
    var leftItems = [],
        rightItems = [];
    if (node.left !== null && parameters.magnitudeMin <= node.midpoint) {
      leftItems = search(node.left, parameters);
    }
    if (node.right !== null && parameters.magnitudeMax > node.midpoint) {
      leftItems = leftItems.concat(search(node.right, parameters));
    }
    return leftItems;
  }
}

function _insert(node, item) {
  if (node.depth < MAX_DEPTH) {
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
  } else {
    if (node.bucket === null) node.bucket = [];
    node.bucket.push(item);
    // Sort is irrelevant at the moment, but could be used for a binary search
    // later.
    //node.bucket.sort((a, b) => { return (a.angle < b.angle) ? -1 : 1; });
  }
}