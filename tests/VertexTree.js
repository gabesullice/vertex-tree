import test from "ava";
import * as vertex from "libshapes/lib/Vertex";
import VertexTree from "../lib/VertexTree";

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
