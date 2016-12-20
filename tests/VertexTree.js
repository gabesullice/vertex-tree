import test from "ava";
import * as vertex from "libshapes/lib/Vertex";
import Shape from "libshapes/lib/Shape";
import {Edge} from "libshapes/lib/Edge";
import {VertexTree,Item} from "../lib/VertexTree";

test("Can create a new VertexTree", t => {
  const vt = new VertexTree();
});

test("Can insert a Vertex into a VertexTree", t => {
  const vt = new VertexTree();
  for (let i = 0; i < 100; i++) {
    const {x, y} = {x: Math.random() * 100, y: Math.random() * 100};
    vt.insert("some-id", new vertex.Vertex(x, y));
  }
});

test("Can insert an item into a VertexTree", t => {
  const vt = new VertexTree();
  vt.insert(vt.newItem("some-id", new vertex.Vertex(0, 0)));
});

test("Can create a new item", t => {
  const vt = new VertexTree();
  const item = vt.newItem("some-id", new vertex.Vertex(0,1));
  t.true(item instanceof Item);
});

test("Can create a new item with edges", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  const item = vt.newItem("some-id", new vertex.Vertex(0,1), {
    edges: shape.edges(),
  });
  t.deepEqual(item.edges, shape.edges());
});

test("Can add an edge to an item", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  const item = vt.newItem("some-id", new vertex.Vertex(0,1), {
    edges: shape.edges(),
  });
  item.addEdge(shape.edges()[0]);
  t.is(item.edges.length, shape.edges().length, "Edges should not be duplicated");

  item.addEdge(new Edge([[0,0], [1,1]]));
  t.is(item.edges.length, shape.edges().length + 1, "A new edge should be inserted");
});

test("Can remove an edge from an item", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  const item = vt.newItem("some-id", new vertex.Vertex(0,1), {
    edges: shape.edges(),
  });
  item.removeEdge(shape.edges()[0]);
  t.is(item.edges.length, shape.edges().length - 1, "An edge should be removed");
});

test("Can query for vertices in a VertexTree", t => {
  const vt = new VertexTree();
  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
  input.forEach((point, index) => {
    vt.insert(index, new vertex.Vertex(point[0], point[1]));
  });
  const cases = [
    {query: {origin: new vertex.Vertex(100, 0), radius: 10}, expected: []},
    {query: {origin: new vertex.Vertex(50, 50), radius: 10}, expected: [[50,50]]},
    {query: {origin: new vertex.Vertex(50, 50), radius: 11}, expected: [[50,40], [50,50]]},
    {query: {origin: new vertex.Vertex(50, 50), radius: 15}, expected: [[50,40], [50,50], [60,60]]},
  ];
  cases.forEach(item => {
    const actual = vt.find(item.query).map(item => {
      return [item.vertex.x, item.vertex.y];
    });
    t.deepEqual(actual, item.expected);
  });
});

test("Can find the nearest vertex in a VertexTree", t => {
  const vt = new VertexTree();
  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
  input.forEach((point, index) => {
    vt.insert(index, new vertex.Vertex(point[0], point[1]));
  });
  const cases = [
    //{search: new vertex.Vertex(100, 0), expected: []},
    {search: new vertex.Vertex(51, 51), expected: [[50,50]]},
    //{search: new vertex.Vertex(50, 50), expected: [[50,40], [50,50]]},
    //{search: new vertex.Vertex(50, 50), expected: [[50,40], [50,50], [60,60]]},
  ];
  cases.forEach(item => {
    const actual = vt.nearest(item.search).map(item => {
      return [item.vertex.x, item.vertex.y];
    });
    t.deepEqual(actual, item.expected);
  });
});

//test("Can add an edge to VertexTree", t => {
//  const shape = new Shape([[0,0], [0,1], [1,0]]);
//  const vt = new VertexTree();
//  const item = vt.newItem("some-id", new vertex.Vertex(0,1), {
//    edges: shape.edges(),
//  });
//
//  const edge = new Edge([[0,0], [1,1]]);
//  vt.addEdge(edge);
//  const result = vt.find({origin: edge.left(), radius: 0.5});
//  t.is(result[0].edges.length, 3, "A new edge should be inserted at 0,0");
//});

//test("Can remove vertices from a VertexTree", t => {
//  const vt = new VertexTree();
//  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
//  input.forEach((point, index) => {
//    vt.insert(index, new vertex.Vertex(point[0], point[1]));
//  });
//  const cases = [
//    {remove: new vertex.Vertex(10, 0),  expected: 6},
//    {remove: new vertex.Vertex(0, 10),  expected: 6},
//    {remove: new vertex.Vertex(50, 50), expected: 6},
//    {remove: new vertex.Vertex(10, 12), expected: 7},
//  ];
//  cases.forEach(item => {
//    vt.remove(item.remove);
//    const result = vt.find({origin: item.remove, radius: 1000}).map(item => {
//      return [item.vertex.x, item.vertex.y];
//    })
//    t.deepEqual(result.length, item.expected);
//  });
//});
