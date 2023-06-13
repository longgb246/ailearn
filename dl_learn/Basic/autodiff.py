import numpy as np


class Node(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_byconst_op(self, other)
        return new_node

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = rdiv_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_byconst_op(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = rsub_byconst_op(self, other)
        return new_node

    def __neg__(self):
        return neg_op(self)

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        assert False, "Implemented in subclass"

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        assert False, "Implemented in subclass"


class NegOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "-%s" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return -input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad]


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]


class SubOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s-%s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, -output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]


class SubByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class RSubByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr - input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]


class DivOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s/%s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[1], -output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1])]


class DivByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "%s/%s" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad / node.const_attr]


class RDivByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "%s/%s" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr / input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad * node.const_attr / (node.inputs[0] * node.inputs[0])]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        """TODO: Your code here"""
        return [output_grad * node.const_attr]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        mat_A = input_vals[0]
        mat_B = input_vals[1]
        if node.matmul_attr_trans_A:
            mat_A = mat_A.T
        if node.matmul_attr_trans_B:
            mat_B = mat_B.T
        return np.matmul(mat_A, mat_B)

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [matmul_op(output_grad, node.inputs[1], False, True),
                matmul_op(node.inputs[0], output_grad, True, False)]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class LogOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "log(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]


class ExpOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "exp(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp_op(node.inputs[0])]


class ReduceSumOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "reduce_sum(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert isinstance(input_vals[0], np.ndarray)
        return np.sum(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * oneslike_op(node.inputs[0])]


# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
div_op = DivOp()
sub_op = SubOp()
neg_op = NegOp()
add_byconst_op = AddByConstOp()
rsub_byconst_op = RSubByConstOp()
sub_byconst_op = SubByConstOp()
mul_byconst_op = MulByConstOp()
div_byconst_op = DivByConstOp()
rdiv_byconst_op = RDivByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
log_op = LogOp()
exp_op = ExpOp()
reduce_sum = ReduceSumOp()


def exp(val):
    if isinstance(val, Node):
        return exp_op(val)
    return np.exp(val)


def log(val):
    if isinstance(val, Node):
        return log_op(val)
    return np.log(val)


class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""

    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.

        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceholderOp):
                continue
            vals = [node_to_val_map[n] for n in node.inputs]
            compute_val = node.op.compute(node, vals)
            node_to_val_map[node] = compute_val if isinstance(
                compute_val, np.ndarray) else np.array(compute_val)

        # Collect node values.
        node_val_results = [node_to_val_map[node]
                            for node in self.eval_node_list]
        return node_val_results


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        for i in range(len(node.inputs)):
            ch = node.inputs[i]
            grads = node.op.gradient(node, grad)
            grads_list = node_to_output_grads_list.get(ch, [])
            grads_list.append(grads[i])
            node_to_output_grads_list[ch] = grads_list

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


##############################
####### Helper Methods #######
##############################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


##############################
#######  Test Methods  #######
##############################


def test_identity():
    x2 = Variable(name="x2")
    y = x2

    grad_x2, = gradients(y, [x2])

    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_add_by_const():
    x2 = Variable(name="x2")
    y = 5 + x2

    grad_x2, = gradients(y, [x2])

    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_sub_by_const():
    x2 = Variable(name='x2')
    y = 3 - x2
    grad_x2, = gradients(y, [x2])
    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, 3 - x2_val)
    assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))


def test_neg():
    x1 = Variable(name='x1')
    x2 = Variable(name='x2')

    y = -x2 + x1

    grad_x1, grad_x2 = gradients(y, [x1, x2])
    executor = Executor([y, grad_x1, grad_x2])
    x2_val = 2 * np.ones(3)
    x1_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, -x2_val + x1_val)
    assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))


def test_mul_by_const():
    x2 = Variable(name="x2")
    y = 5 * x2

    grad_x2, = gradients(y, [x2])

    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)


def test_div_two_vars():
    x1 = Variable(name='x1')
    x2 = Variable(name='x2')

    y = x1 / x2

    grad_x1, grad_x2 = gradients(y, [x1, x2])

    executor = Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 5 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val / x2_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) / x2_val)
    assert np.array_equal(grad_x2_val, -x1_val / (x2_val * x2_val))


def test_div_by_const():
    x2 = Variable(name="x2")
    y = 5 / x2

    grad_x2, = gradients(y, [x2])

    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, 5 / x2_val)
    print(grad_x2_val)
    print(-5 / (x2_val * x2_val))
    assert np.array_equal(grad_x2_val, -5 / (x2_val * x2_val))


def test_add_two_vars():
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    y = x2 + x3

    grad_x2, grad_x3 = gradients(y, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val + x3_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_x3_val, np.ones_like(x3_val))


def test_mul_two_vars():
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    y = x2 * x3

    grad_x2, grad_x3 = gradients(y, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():
    x1 = Variable(name="x1")
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    y = x1 + x2 * x3 * x1

    grad_x1, grad_x2, grad_x3 = gradients(y, [x1, x2, x3])

    executor = Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val, x3: x3_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val * x1_val)
    assert np.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():
    x1 = Variable(name="x1")
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    x4 = Variable(name="x4")
    y = x1 + x2 * x3 * x4

    grad_x1, grad_x2, grad_x3, grad_x4 = gradients(y, [x1, x2, x3, x4])

    executor = Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3

    grad_x2, grad_x3 = gradients(y, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val})

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * \
        (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_grad_of_grad():
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    y = x2 * x2 + x2 * x3

    grad_x2, grad_x3 = gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = gradients(grad_x2, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val})

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)


def test_matmul_two_vars():
    x2 = Variable(name="x2")
    x3 = Variable(name="x3")
    y = matmul_op(x2, x3)

    grad_x2, grad_x3 = gradients(y, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(
        feed_dict={x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(
        np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(
        np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_log_op():
    x1 = Variable(name="x1")
    y = log(x1)

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.log(x1_val))
    assert np.array_equal(grad_x1_val, 1 / x1_val)


def test_log_two_vars():
    x1 = Variable(name="x1")
    x2 = Variable(name="x2")
    y = log(x1 * x2)

    grad_x1, grad_x2 = gradients(y, [x1, x2])

    executor = Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.log(x1_val * x2_val))
    assert np.array_equal(grad_x1_val, x2_val / (x1_val * x2_val))
    assert np.array_equal(grad_x2_val, x1_val / (x1_val * x2_val))


def test_exp_op():
    x1 = Variable(name="x1")
    y = exp(x1)

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.exp(x1_val))
    assert np.array_equal(grad_x1_val, np.exp(x1_val))


def test_exp_mix_op():
    x1 = Variable(name="x1")
    x2 = Variable(name="x2")
    y = exp(log(x1 * x2) + 1)

    grad_x1, grad_x2 = gradients(y, [x1, x2])

    executor = Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(
        feed_dict={x1: x1_val, x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.exp(np.log(x1_val * x2_val) + 1))
    assert np.array_equal(grad_x1_val, y_val * x2_val / (x1_val * x2_val))
    assert np.array_equal(grad_x2_val, y_val * x1_val / (x1_val * x2_val))


def test_reduce_sum():
    x1 = Variable(name="x1")
    y = reduce_sum(x1)

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.sum(x1_val))
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))


def test_reduce_sum_mix():
    x1 = Variable(name="x1")
    y = exp(reduce_sum(x1))

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})
    expected_y_val = np.exp(np.sum(x1_val))
    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.array_equal(grad_x1_val, expected_y_val * np.ones_like(x1_val))

    y2 = log(reduce_sum(x1))
    grad_x2, = gradients(y2, [x1])
    executor2 = Executor([y2, grad_x2])
    y2_val, grad_x2_val = executor2.run(feed_dict={x1: x1_val})
    expected_y2_val = np.log(np.sum(x1_val))
    assert isinstance(y2, Node)
    assert np.array_equal(y2_val, expected_y2_val)
    assert np.array_equal(grad_x2_val, (1/np.sum(x1_val))
                          * np.ones_like(x1_val))


def test_mix_all():
    x1 = Variable(name="x1")
    y = 1/(1+exp(-reduce_sum(x1)))

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val = executor.run(feed_dict={x1: x1_val})
    expected_y_val = 1/(1+np.exp(-np.sum(x1_val)))
    expected_y_grad = expected_y_val * \
        (1 - expected_y_val) * np.ones_like(x1_val)

    print(expected_y_grad)
    print(grad_x1_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_x1_val - expected_y_grad)) < 1E-10


def test_logistic():
    x1 = Variable(name="x1")
    w = Variable(name='w')
    y = 1/(1+exp(-reduce_sum(w * x1)))

    grad_w, = gradients(y, [w])

    executor = Executor([y, grad_w])
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
    expected_y_val = 1/(1 + np.exp(-np.sum(w_val * x1_val)))
    expected_y_grad = expected_y_val * (1 - expected_y_val) * x1_val

    print(expected_y_grad)
    print(grad_w_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7


def test_log_logistic():
    x1 = Variable(name="x1")
    w = Variable(name='w')
    y = log(1/(1+exp(-reduce_sum(w * x1))))

    grad_w, = gradients(y, [w])

    executor = Executor([y, grad_w])
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
    logistic = 1/(1+np.exp(-np.sum(w_val * x1_val)))
    expected_y_val = np.log(logistic)
    expected_y_grad = (1 - logistic) * x1_val

    print(expected_y_grad)
    print(grad_w_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7


def test_logistic_loss():
    x = Variable(name='x')
    w = Variable(name='w')
    y = Variable(name='y')

    h = 1 / (1 + exp(-reduce_sum(w * x)))
    L = y * log(h) + (1 - y) * log(1 - h)
    w_grad, = gradients(L, [w])
    executor = Executor([L, w_grad])

    y_val = 0
    x_val = np.array([2, 3, 4])
    w_val = np.random.random(3)

    L_val, w_grad_val = executor.run(feed_dict={x: x_val, y: y_val, w: w_val})

    logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
    expected_L_val = y_val * np.log(logistic) + \
        (1 - y_val) * np.log(1 - logistic)
    expected_w_grad = (y_val - logistic) * x_val

    print(expected_L_val)
    print(L_val)
    print(expected_w_grad)
    print(w_grad_val)

    assert expected_L_val == L_val
    assert np.sum(np.abs(expected_w_grad - w_grad_val)) < 1E-9


##############################
#######  lr_autodiff   #######
##############################


def logistic_prob(_w):
    def wrapper(_x):
        return 1 / (1 + np.exp(-np.sum(_x * _w)))
    return wrapper


def test_accuracy(_w, _X, _Y):
    prob = logistic_prob(_w)
    correct = 0
    total = len(_Y)
    for i in range(len(_Y)):
        x = _X[i]
        y = _Y[i]
        p = prob(x)
        if p >= 0.5 and y == 1.0:
            correct += 1
        elif p < 0.5 and y == 0.0:
            correct += 1
    print("总数：%d, 预测正确：%d" % (total, correct))


def plot(N, X_val, Y_val, w, with_boundary=False):
    import matplotlib.pyplot as plt
    for i in range(N):
        __x = X_val[i]
        if Y_val[i] == 1:
            plt.plot(__x[1], __x[2], marker='x')
        else:
            plt.plot(__x[1], __x[2], marker='o')
    if with_boundary:
        min_x1 = min(X_val[:, 1])
        max_x1 = max(X_val[:, 1])
        min_x2 = float(-w[0] - w[1] * min_x1) / w[2]
        max_x2 = float(-w[0] - w[1] * max_x1) / w[2]
        plt.plot([min_x1, max_x1], [min_x2, max_x2], '-r')

    plt.show()


def gen_2d_data(n):
    x_data = np.random.random([n, 2])
    y_data = np.ones(n)
    for i in range(n):
        d = x_data[i]
        if d[0] + d[1] < 1:
            y_data[i] = 0
    x_data_with_bias = np.ones([n, 3])
    x_data_with_bias[:, 1:] = x_data
    return x_data_with_bias, y_data


def auto_diff_lr():
    x = Variable(name='x')
    w = Variable(name='w')
    y = Variable(name='y')

    # 注意，以下实现某些情况会有很大的数值误差，
    # 所以一般真实系统实现会提供高阶算子，从而减少数值误差

    h = 1 / (1 + exp(-reduce_sum(w * x)))
    L = y * log(h) + (1 - y) * log(1 - h)
    w_grad, = gradients(L, [w])
    executor = Executor([L, w_grad])

    N = 100
    X_val, Y_val = gen_2d_data(N)
    w_val = np.ones(3)

    plot(N, X_val, Y_val, w_val)
    executor = Executor([L, w_grad])
    test_accuracy(w_val, X_val, Y_val)
    alpha = 0.01
    max_iters = 300
    for iteration in range(max_iters):
        acc_L_val = 0
        for i in range(N):
            x_val = X_val[i]
            y_val = np.array(Y_val[i])
            L_val, w_grad_val = executor.run(
                feed_dict={w: w_val, x: x_val, y: y_val})
            w_val += alpha * w_grad_val
            acc_L_val += L_val
        print("iter = %d, likelihood = %s, w = %s" %
              (iteration, acc_L_val, w_val))
    test_accuracy(w_val, X_val, Y_val)
    plot(N, X_val, Y_val, w_val, True)


# 这一生关于你的风景
# 无念
# 56 - 1
# 78 - 2
# 910 - 3
