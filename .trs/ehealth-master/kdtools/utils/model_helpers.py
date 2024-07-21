from pydot import Dot, Node, Edge

class Tree(object):
    def __init__(self, idx: int, info: str):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.idx = idx
        self.state = None
        self.info = info

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', None):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth', None):
            return self._depth
        self._depth = self.parent.depth() + 1 if self.parent is not None else 0
        return self._depth

    def height(self):
        if getattr(self, '_height', None):
            return self._height
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_height = self.children[i].height()
                if child_height > count:
                    count = child_height
            count += 1
        self._height = count
        return self._height

    @property
    def nodes(self):
        if getattr(self, '_nodes', None):
            return self._nodes

        self._nodes = [self]
        for child in self.children:
            self._nodes.extend(child.nodes)

        return self._nodes

    @property
    def edges(self):
        if getattr(self, '_edges', None):
            return self._edges

        self._edges = [(self, child) for child in self.children]
        for child in self.children:
            self._edges.extend(child.edges)

        return self._edges

    @staticmethod
    def _lca(n1, n2):
        if n1.depth() <= n2.depth():
            n1, n2 = n2, n1

        while n1.depth() > n2.depth():
            n1 = n1.parent

        while n1.idx != n2.idx:
            n1 = n1.parent
            n2 = n2.parent

        return n1

    @staticmethod
    def _path2ancestor(n, ancestor, reverse = False):
        path = []
        while n.idx != ancestor.idx:
            path.append(n)
            n = n.parent
        if reverse:
            path.reverse()
        return path

    def path(n1, n2):
        lca = Tree._lca(n1, n2)
        n12lca = Tree._path2ancestor(n1, lca)
        lca2n2 = Tree._path2ancestor(n2, lca, reverse = True)

        return n12lca + [lca] + lca2n2

    def pydot_img(self, path: str):
        graph = Dot(graph_type='digraph')
        graph.set_node_defaults(
            color='lightgray',
            style='filled',
            shape='box',
            fontname='Courier',
            fontsize='10'
        )

        nodes = {node.idx:Node(f"{node.info}") for node in self.nodes}
        for u,v in self.edges:
            graph.add_edge(Edge(nodes[u.idx], nodes[v.idx]))

        p = r"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz\dot.exe"

        graph.write_png(path, prog = p)
