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
    vt.insert(new vertex.Vertex(x, y));
  }
});

test("Can insert an item into a VertexTree", t => {
  const vt = new VertexTree();
  vt.insert(vt.newItem(new vertex.Vertex(0, 0)));
});

test("Can create a new item", t => {
  const vt = new VertexTree();
  const item = vt.newItem(new vertex.Vertex(0,1));
  t.true(item instanceof Item);
});

test("Can create a new item with edges", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  const item = vt.newItem(new vertex.Vertex(0,1), {
    edges: shape.edges(),
  });
  t.deepEqual(item.edges, shape.edges());
});

test("Can add an edge to an item", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  const item = vt.newItem(new vertex.Vertex(0,1), {
    edges: shape.edges(),
  });
  item.addEdge(shape.edges()[0]);
  t.is(item.edges.length, shape.edges().length + 1, "Edges should not be deduplicated");

  item.addEdge(new Edge([[0,0], [1,1]]));
  t.is(item.edges.length, shape.edges().length + 2, "A new edge should be inserted");
});

test("Can remove an edges from a VertexTree", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  shape.edges().forEach(edge => {
    vt.insertEdge(edge);
  });
  const item = vt.at(shape.vertices()[0]);
  t.is(item.edges.length, 2, "Edges should be added");
  vt.removeEdge(shape.edges()[0]);
  t.is(item.edges.length, 1, "An edge should be removed");

  vt.insertEdge(shape.edges()[0]); // add removed back in
  vt.insertEdge(shape.edges()[0]); // add it again
  vt.removeEdge(shape.edges()[0]); // remove it once
  t.is(item.edges.length, 2, "Only one edge should be removed at a time");
});

test("Removing the last edge from an item removes the item", t => {
  const shape = new Shape([[0,0], [0,1], [1,0]]);
  const vt = new VertexTree();
  shape.edges().forEach(edge => {
    vt.insertEdge(edge);
  });
  let item = vt.at(shape.vertices()[0]);
  t.truthy(item, "There should be an item at (0,0)");
  vt.removeEdge(shape.edges()[0]);
  t.truthy(vt.at(shape.vertices()[0]), "Removing one incoming edge to (0,0) should not remove the item");
  vt.removeEdge(shape.edges()[2]);
  t.falsy(vt.at(shape.vertices()[0]), "Removing the incoming edges to (0,0) should remove the item");
});

test("Can query for vertices in a VertexTree", t => {
  const vt = new VertexTree();
  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
  input.forEach((point, index) => {
    vt.insert(new vertex.Vertex(point[0], point[1]));
  });
  const cases = [
    {query: {origin: new vertex.Vertex(100, 0), radius: 10}, expected: []},
    {query: {origin: new vertex.Vertex(50, 50), radius: 10}, expected: [[50,50]]},
    {query: {origin: new vertex.Vertex(50, 50), radius: 11}, expected: [[50,40], [50,50]]},
    {query: {origin: new vertex.Vertex(50, 50), radius: 15}, expected: [[50,40], [50,50], [60,60]]},
    {query: {origin: new vertex.Vertex(0, 0), radius: Math.sqrt(Math.pow(10, 2) * 2)}, expected: [[10,0], [0,10]]},
    {query: {origin: new vertex.Vertex(0, 0), radius: Math.sqrt(Math.pow(10, 2) * 2) * 1.001}, expected: [[10,0], [0,10], [10,10]]},
  ];
  cases.forEach(item => {
    const actual = vt.find(item.query).map(item => {
      return [item.vertex.x, item.vertex.y];
    });
    t.deepEqual(actual, item.expected);
  });
});

test("Can get an item by its vertex", t => {
  const vt = new VertexTree();
  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
  input.forEach((point, index) => {
    vt.insert(new vertex.Vertex(point[0], point[1]));
  });
  const cases = [
    {query: [10,0], expect: "object"},
    {query: [10,50], expect: "object"},
    {query: [0,0], expect: "undefined"},
    {query: [50,40], expect: "object"},
    {query: [33,33], expect: "undefined"},
  ];
  cases.forEach(item => {
    const point = new vertex.Vertex(item.query[0], item.query[1]);
    const found = vt.at(point);
    t.is(typeof(found), item.expect);
  });
});

test("Can add an edge to VertexTree", t => {
  const vt = new VertexTree();
  const cases = [
    {insert: [[0,0], [0,1]], subtests: [
      {query: [0,1], expect: 1, label: "A new edge should be inserted at 0,1"},
    ]},
    {insert: [[0,1], [1,0]], subtests: [
      {query: [1,0], expect: 1, label: "A new edge should be inserted at 1,0"},
      {query: [0,1], expect: 2, label: "Two edges should eminate from 0,1"},
    ]},
    {insert: [[1,0], [0,0]], subtests: [
      {query: [0,0], expect: 2, label: "Two edges should eminate from 0,0"},
    ]},
    {insert: [[0,1], [1,1]], subtests: [
      {query: [0,1], expect: 3, label: "Three edges should eminate from 0,1"},
    ]},
    {insert: [[1,1], [1,0]], subtests: [
      {query: [1,0], expect: 3, label: "Three edges should eminate from 1,0"},
    ]},
  ];
  cases.forEach(item => {
    vt.insertEdge(new Edge(item.insert));
    item.subtests.forEach(sub => {
      t.is(
        vt.at(new vertex.Vertex(sub.query[0], sub.query[1])).edges.length, 
        sub.expect,
        sub.label
      );
    });
  });
});

test("Can remove vertices from a VertexTree", t => {
  const vt = new VertexTree();
  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
  input.forEach((point, index) => {
    vt.insert(new vertex.Vertex(point[0], point[1]));
  });
  const cases = [
    {remove: [10,0], subtests: [
      {query: [10,0], expect: "falsy"},
      {query: [10,50], expect: "truthy"},
      {query: [0,10], expect: "truthy"},
    ]},
    {remove: [0,10], subtests: [{query: [0,10], expect: "falsy"}]},
    {remove: [50,50], subtests: [{query: [50,50], expect: "falsy"}]},
    {remove: [10,12], subtests: [
      {query: [10,12], expect: "falsy"},
      {query: [60,60], expect: "truthy"},
    ]},
  ];
  cases.forEach(item => {
    vt.remove(new vertex.Vertex(item.remove[0], item.remove[1]));
    item.subtests.forEach(sub => {
      const point = new vertex.Vertex(sub.query[0], sub.query[1]);
      const found = vt.at(point);
      switch (sub.expect) {
        case "truthy":
          t.truthy(found);
          break;
        case "falsy":
          t.falsy(found);
          break;
      }
    });
  });
});

test("Can remove an edge from a VertexTree", t => {
  const vt = new VertexTree();
  const input = [
    [[0,0], [0,10]],
    [[0,10], [10,10]],
    [[10,10], [10,0]],
    [[10,0], [0,0]],
  ];
  input.forEach(edge => {
    vt.insertEdge(new Edge(edge));
  });
  const cases = [
    {remove: [], subtests: [{query: [0,0], length: 2}]},
    {remove: [[[0,0], [0,10]]], subtests: [
      {query: [0,0], length: 1},
      {query: [0,10], length: 1},
    ], didRemove: true},
    {remove: [[[10,0], [0,0]]], subtests: [
      {query: [0,0], length: 0},
      {query: [10,0], length: 1},
    ], didRemove: true},
    {remove: [], subtests: [{query: [20,20], length: 0}]},
    {remove: [[[30,30], [50,50]]], subtests: [{query: [20,20], length: 0}], didRemove: false},
  ];
  cases.forEach(item => {
    item.remove.forEach(edge => {
      const result = vt.removeEdge(new Edge(edge));
      t.is(result, item.didRemove);
    });
    item.subtests.forEach(sub => {
      const found = vt.at(new vertex.Vertex(sub.query[0], sub.query[1]));
      if (found) {
        t.is(found.edges.length, sub.length);
      }
    });
  });
});

test("Can create new Items with tags", t => {
  const vt = new VertexTree();
  const tags = ["monkey"];
  const item  = vt.newItem(new vertex.Vertex(0,0), { tags });
  t.deepEqual(item.tags, tags);
});

test("Can insert edges with tags", t => {
  const vt = new VertexTree();
  const tags = ["monkey"];
  vt.insertEdge(new Edge([[0,0],[10,10]]), tags);
  t.deepEqual(vt.at(new vertex.Vertex(0,0)).tags, tags);
});

test("Can remove tags from an item", t => {
  const vt = new VertexTree();
  const tags = ["monkey"];
  const item  = vt.newItem(new vertex.Vertex(0,0), { tags });
  item.removeTag("monkey");
  t.deepEqual(item.tags, []);
});

//test("Can find the nearest vertex in a VertexTree", t => {
//  const vt = new VertexTree();
//  const input = [[10,0], [0,10], [10, 50], [10,10], [50,50], [60,60], [50,40]];
//  input.forEach((point, index) => {
//    vt.insert(new vertex.Vertex(point[0], point[1]));
//  });
//  const cases = [
//    {search: new vertex.Vertex(59, 59), expected: [[60,60]]},
//    //{search: new vertex.Vertex(51, 51), expected: [[50,50]]},
//    //{search: new vertex.Vertex(0, 1), expected: [[0,10]]},
//  ];
//  cases.forEach(item => {
//    const actual = vt.nearest(item.search).map(item => {
//      console.log(item);
//      return [item.vertex.x, item.vertex.y];
//    });
//    t.deepEqual(actual, item.expected);
//  });
//});
