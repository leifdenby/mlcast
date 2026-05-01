#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = ["graphviz", "loguru"]
# ///

# Standard library
import argparse
import ast
import collections
import os
import sys
import xml.etree.ElementTree as ET

# Third-party
import graphviz
from loguru import logger

DRAWIO_CONTAINER_LABEL_WIDTH = 44
DRAWIO_CONTAINER_INSET_X = 56
DRAWIO_NODE_WIDTH = 220
DRAWIO_NODE_HEIGHT = 56
DRAWIO_NODE_X_GAP = 20
DRAWIO_NODE_Y_GAP = 16
DRAWIO_EDGE_STYLE = "edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;jettySize=auto;rounded=0;html=1;"


RELEVANT_EXTERNAL_BASE_NAMES = {
    "Dataset",
    "GroupNorm",
    "LightningDataModule",
    "LightningModule",
    "Module",
    "ModuleDict",
    "Sequential",
}

RELEVANT_EXTERNAL_INSTANTIATION_NAMES = {
    "Trainer",
}

EXPLICIT_RELEVANT_CLASSES = {
    "LDCast",
    "PLMSSampler",
}


def cluster_name_from_module(module_name):
    return f"cluster_{module_name.replace('.', '_')}"


def cluster_anchor_name(module_name):
    return f"anchor_{module_name.replace('.', '_')}"


class ClassHierarchyExtractor(ast.NodeVisitor):
    def __init__(self, base_module, global_class_hierarchy):
        self.class_hierarchy = global_class_hierarchy  # Use global hierarchy to ensure complete detection
        self.class_modules = {}  # Dictionary to store class to module mapping
        self.external_inheritance = []  # List to store external class inheritance
        self.instantiations = []  # List to store class instantiations
        self.external_instantiations = []  # Relevant external classes instantiated by tracked classes
        self.references = []  # List to store class references (e.g., via dataclass fields)
        self.memberships = []  # List to store class memberships via type annotations
        self.current_class = None  # Track the current class
        self.imports = {}  # Track imports for resolving external class references
        self.abc_classes = set()  # Names of classes detected as abstract
        self.base_module = base_module

    def first_pass(self, tree, file_path):
        # Build a dotted module path relative to the base package, e.g.
        # ".../neural_lam/models/base_graph_model.py" -> "models.base_graph_model"
        no_ext = os.path.splitext(file_path)[0]
        parts = no_ext.replace(os.sep, "/").split("/")
        base_pkg = self.base_module.rstrip(".")
        if base_pkg in parts:
            idx = len(parts) - 1 - parts[::-1].index(base_pkg)
            rel_parts = parts[idx + 1 :]
            module_path = ".".join(rel_parts)
        else:
            module_path = ".".join(parts[-2:])

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.class_hierarchy[node.name] = []
                self.class_modules[node.name] = module_path

    def second_pass(self, tree):
        self.visit(tree)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}" if node.module else alias.name

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ClassDef(self, node):
        self.current_class = node.name
        parents = []

        # Detect inheritance from abc.ABC, bare ABC, or metaclass=ABCMeta
        is_abc = False
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ("ABC", "ABCMeta"):
                    is_abc = True
                elif base.id in self.class_hierarchy:
                    parents.append(base.id)
                else:
                    resolved_external = self.resolve_imported_symbol(base.id)
                    if resolved_external:
                        self.external_inheritance.append((self.current_class, resolved_external))
                    else:
                        parents.append(base.id)
            elif isinstance(base, ast.Attribute):
                external_name = self.get_full_attribute_name(base)
                if external_name in ("abc.ABC", "abc.ABCMeta"):
                    is_abc = True
                else:
                    self.external_inheritance.append((self.current_class, external_name))
        for kw in node.keywords:
            if kw.arg == "metaclass":
                meta_name = self.get_class_name_from_node(kw.value)
                if meta_name in ("ABCMeta",):
                    is_abc = True

        # Detect @abstractmethod decorators on any method body
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                for dec in stmt.decorator_list:
                    dec_name = self.get_class_name_from_node(dec)
                    if dec_name in ("abstractmethod", "abstractproperty"):
                        is_abc = True
                        break
            if is_abc:
                break

        if is_abc:
            self.abc_classes.add(self.current_class)

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                self._record_membership_from_annotation(stmt.annotation)
            elif isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                self._inspect_init_params(stmt)
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                self._scan_method_for_instantiations(stmt)

        self.class_hierarchy[self.current_class] = parents
        self.generic_visit(node)
        self.current_class = None

    def _record_membership_from_annotation(self, annotation):
        if annotation is None:
            return
        for member_class in self._get_annotation_class_names(annotation):
            if member_class and member_class in self.class_hierarchy:
                edge = (self.current_class, member_class)
                if edge not in self.memberships:
                    self.memberships.append(edge)
                    logger.debug(f"Detected membership: {self.current_class} -> {member_class}")

    def _get_annotation_class_names(self, annotation):
        if isinstance(annotation, ast.Name | ast.Attribute):
            class_name = self.get_class_name_from_node(annotation)
            return [class_name] if class_name else []
        if isinstance(annotation, ast.Subscript):
            return self._get_annotation_class_names(annotation.slice)
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return self._get_annotation_class_names(annotation.left) + self._get_annotation_class_names(
                annotation.right
            )
        if isinstance(annotation, ast.Tuple):
            class_names = []
            for elt in annotation.elts:
                class_names.extend(self._get_annotation_class_names(elt))
            return class_names
        return []

    def _inspect_init_params(self, init_node):
        # Record memberships from __init__ parameter annotations
        for arg in list(init_node.args.args) + list(init_node.args.kwonlyargs):
            self._record_membership_from_annotation(arg.annotation)

    def _scan_method_for_instantiations(self, func_node):
        # Walk all Call nodes inside the method body and treat any call whose
        # target resolves to a tracked class as an instantiation by the
        # enclosing class. This catches:
        #   - self.x = Cls(...)            (assigned to self in any method)
        #   - tmp = Cls(...)               (assigned to local)
        #   - Outer(Inner(...))            (nested in another call)
        #   - nn.ModuleList([Cls(), ...])  (inside container constructors)
        #   - return Cls(...)              (returned without assignment)
        for sub in ast.walk(func_node):
            if not isinstance(sub, ast.Call):
                continue
            class_name = self.get_class_name_from_node(sub.func)
            raw_class_name = self.get_full_attribute_name(sub.func) or class_name
            if not class_name:
                continue
            resolved_class = self.resolve_imported_class(class_name)
            if resolved_class and resolved_class in self.class_hierarchy and resolved_class != self.current_class:
                edge = (self.current_class, resolved_class)
                if edge not in self.instantiations:
                    self.instantiations.append(edge)
                    logger.info(f"Detected instantiation in {func_node.name}: {self.current_class} -> {resolved_class}")
                continue
            self._record_external_instantiation(class_name, raw_class_name, func_node.name)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and self.current_class:
            class_name = self.get_class_name_from_node(node.value.func)
            raw_class_name = self.get_full_attribute_name(node.value.func) or class_name
            logger.debug(f"Assignment detected in class '{self.current_class}': class_name='{class_name}'")
            resolved_class = self.resolve_imported_class(class_name)
            logger.debug(f"Resolved class: '{resolved_class}'")
            if resolved_class in self.class_hierarchy and resolved_class != self.current_class:
                edge = (self.current_class, resolved_class)
                if edge not in self.instantiations:
                    logger.info(f"Detected instantiation: {self.current_class} -> {resolved_class}")
                    self.instantiations.append(edge)
            else:
                self._record_external_instantiation(class_name, raw_class_name)
        self.generic_visit(node)

    def _record_external_instantiation(self, class_name, raw_class_name="", context_name=None):
        if not class_name or not self.current_class:
            return
        resolved_name = self.resolve_relevant_external_class(class_name, raw_class_name)
        if not resolved_name:
            return
        edge = (self.current_class, resolved_name)
        if edge in self.external_instantiations:
            return
        self.external_instantiations.append(edge)
        if context_name:
            logger.info(f"Detected external instantiation in {context_name}: {self.current_class} -> {resolved_name}")
        else:
            logger.info(f"Detected external instantiation: {self.current_class} -> {resolved_name}")

    def resolve_imported_class(self, class_name):
        if class_name in self.class_hierarchy:
            return class_name
        for _imported_name, full_name in self.imports.items():
            if full_name.split(".")[-1] == class_name:
                return full_name.split(".")[-1]
        return ""

    def resolve_relevant_external_class(self, class_name, raw_class_name=""):
        resolved_symbol = self.resolve_imported_symbol(raw_class_name)
        if resolved_symbol and resolved_symbol.split(".")[-1] in RELEVANT_EXTERNAL_INSTANTIATION_NAMES:
            return resolved_symbol
        for full_name in self.imports.values():
            if full_name.split(".")[-1] != class_name:
                continue
            if class_name in RELEVANT_EXTERNAL_INSTANTIATION_NAMES:
                return full_name
        return class_name if class_name in RELEVANT_EXTERNAL_INSTANTIATION_NAMES else ""

    def resolve_imported_symbol(self, symbol_name):
        if not symbol_name:
            return ""
        if symbol_name in self.imports:
            return self.imports[symbol_name]
        if "." not in symbol_name:
            return self.imports.get(symbol_name, "")
        root_name, _, remainder = symbol_name.partition(".")
        imported_root = self.imports.get(root_name)
        if not imported_root:
            return ""
        return f"{imported_root}.{remainder}" if remainder else imported_root

    def get_class_name_from_node(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_full_attribute_name(node).split(".")[-1]
        return ""

    def get_full_attribute_name(self, node):
        if isinstance(node, ast.Attribute):
            return f"{self.get_full_attribute_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        return ""


def extract_class_hierarchy_from_paths(paths, base_module):
    global_class_hierarchy = {}
    modules = {}
    instantiations = []
    external_inheritance = []
    external_instantiations = []
    references = []
    memberships = []
    abc_classes = set()
    trees = []
    files = []

    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, file_list in os.walk(path):
                for file in file_list:
                    if file.endswith(".py"):
                        files.append(os.path.join(root, file))

    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
            trees.append((file_path, tree))
            extractor = ClassHierarchyExtractor(base_module, global_class_hierarchy)
            extractor.first_pass(tree, file_path)
            modules.update(extractor.class_modules)

    for _file_path, tree in trees:
        extractor = ClassHierarchyExtractor(base_module, global_class_hierarchy)
        extractor.second_pass(tree)
        for edge in extractor.instantiations:
            if edge not in instantiations:
                instantiations.append(edge)
        for edge in extractor.external_inheritance:
            if edge not in external_inheritance:
                external_inheritance.append(edge)
        for edge in extractor.external_instantiations:
            if edge not in external_instantiations:
                external_instantiations.append(edge)
        for edge in extractor.references:
            if edge not in references:
                references.append(edge)
        for edge in extractor.memberships:
            if edge not in memberships:
                memberships.append(edge)
        abc_classes.update(extractor.abc_classes)

    return (
        global_class_hierarchy,
        modules,
        instantiations,
        external_inheritance,
        external_instantiations,
        references,
        memberships,
        abc_classes,
    )


def select_relevant_classes(class_hierarchy, class_modules, external_inheritance):
    relevant_classes = set(EXPLICIT_RELEVANT_CLASSES)
    pending = True

    while pending:
        pending = False
        for cls, parents in class_hierarchy.items():
            if cls in relevant_classes:
                continue

            if any(parent in relevant_classes for parent in parents):
                relevant_classes.add(cls)
                pending = True
                continue

            external_parents = [parent for child, parent in external_inheritance if child == cls]
            if any(parent.split(".")[-1] in RELEVANT_EXTERNAL_BASE_NAMES for parent in external_parents):
                relevant_classes.add(cls)
                pending = True

    return {
        cls for cls in relevant_classes if cls in class_hierarchy and class_modules.get(cls, "").startswith("models")
    } | {
        cls
        for cls in relevant_classes
        if cls in class_hierarchy and class_modules.get(cls, "") == "data.zarr_datamodule"
    }


def assign_classes_to_clusters(relevant_classes, class_modules, cluster_modules):
    if not cluster_modules:
        return {}, set()

    clusters = {module_name: [] for module_name in cluster_modules}
    matched_modules = set()

    for cls in sorted(relevant_classes):
        module_name = class_modules.get(cls, "")
        matches = [
            cluster_module
            for cluster_module in cluster_modules
            if module_name == cluster_module or module_name.startswith(f"{cluster_module}.")
        ]
        if not matches:
            continue

        best_match = max(matches, key=len)
        clusters[best_match].append(cls)
        matched_modules.add(best_match)

    return clusters, matched_modules


def group_cluster_classes_by_submodule(cluster_module, cluster_classes, class_modules):
    grouped_classes = {}
    direct_classes = []

    for cls in cluster_classes:
        module_name = class_modules.get(cls, "")
        suffix = module_name.removeprefix(cluster_module).lstrip(".")
        if not suffix:
            direct_classes.append(cls)
            continue

        next_segment = suffix.split(".", 1)[0]
        submodule_name = f"{cluster_module}.{next_segment}"
        grouped_classes.setdefault(submodule_name, []).append(cls)

    return direct_classes, grouped_classes


def get_diagram_scope(
    class_hierarchy,
    class_modules,
    external_inheritance,
    cluster_modules,
):
    relevant_classes = select_relevant_classes(class_hierarchy, class_modules, external_inheritance)
    clusters, matched_cluster_modules = assign_classes_to_clusters(relevant_classes, class_modules, cluster_modules)

    for cluster_module in cluster_modules:
        if cluster_module not in matched_cluster_modules:
            logger.warning(f"Requested cluster module '{cluster_module}' matched no classes.")

    return relevant_classes, clusters


def build_layout_edges(class_hierarchy, instantiations, memberships, relevant_classes):
    edges = []

    for cls, parents in class_hierarchy.items():
        if cls not in relevant_classes:
            continue
        for parent in parents:
            if parent != "ABC" and parent in relevant_classes:
                edges.append((parent, cls))

    for source, target in instantiations:
        if source not in relevant_classes or target not in relevant_classes:
            continue
        if (source, target) in memberships:
            continue
        edges.append((target, source))

    for source, target in memberships:
        if source not in relevant_classes or target not in relevant_classes:
            continue
        edges.append((target, source))

    return edges


def compute_class_ranks(relevant_classes, layout_edges):
    outgoing = {cls: set() for cls in relevant_classes}
    indegree = {cls: 0 for cls in relevant_classes}

    for source, target in layout_edges:
        if target in outgoing[source]:
            continue
        outgoing[source].add(target)
        indegree[target] += 1

    queue = collections.deque(sorted(cls for cls, degree in indegree.items() if degree == 0))
    ranks = {cls: 0 for cls in relevant_classes}
    visited = 0

    while queue:
        current = queue.popleft()
        visited += 1
        for child in sorted(outgoing[current]):
            ranks[child] = max(ranks[child], ranks[current] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if visited != len(relevant_classes):
        # Fall back deterministically for cycles: push remaining cyclic nodes below
        # the highest resolved rank of any incoming neighbor we have seen.
        unresolved = sorted(cls for cls, degree in indegree.items() if degree > 0)
        for cls in unresolved:
            incoming_ranks = [ranks[source] for source, target in layout_edges if target == cls and source in ranks]
            ranks[cls] = (max(incoming_ranks) + 1) if incoming_ranks else 0

    return ranks


def write_drawio_diagram(
    class_hierarchy,
    class_modules,
    instantiations,
    external_inheritance,
    external_instantiations,
    memberships,
    output_file,
    abc_classes=None,
    cluster_modules=None,
    cluster_on_submodule_by=3,
):
    abc_classes = abc_classes or set()
    cluster_modules = cluster_modules or []
    relevant_classes, clusters = get_diagram_scope(
        class_hierarchy, class_modules, external_inheritance, cluster_modules
    )
    layout_edges = build_layout_edges(class_hierarchy, instantiations, memberships, relevant_classes)
    class_ranks = compute_class_ranks(relevant_classes, layout_edges)
    clustered_classes = {cls for grouped_classes in clusters.values() for cls in grouped_classes}
    unclustered_classes = sorted(relevant_classes - clustered_classes)

    root = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17")
    diagram = ET.SubElement(root, "diagram", name="Class Diagram")
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1600",
        dy="900",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="1920",
        pageHeight="1080",
        math="0",
        shadow="0",
    )
    root_cell = ET.SubElement(model, "root")
    ET.SubElement(root_cell, "mxCell", id="0")
    ET.SubElement(root_cell, "mxCell", id="1", parent="0")

    next_id = 2
    node_ids = {}
    external_y = 20

    def alloc_id():
        nonlocal next_id
        current = str(next_id)
        next_id += 1
        return current

    def add_geometry(cell, x, y, width, height, relative="0"):
        ET.SubElement(
            cell,
            "mxGeometry",
            attrib={
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "relative": relative,
                "as": "geometry",
            },
        )

    def add_container(parent_id, label, x, y, width, height, dashed=False):
        style = (
            "swimlane;fontStyle=1;horizontal=0;"
            f"startSize={DRAWIO_CONTAINER_LABEL_WIDTH};rounded=1;"
            "container=1;collapsible=0;whiteSpace=wrap;html=1;"
        )
        if dashed:
            style += "dashed=1;"
        cell = ET.SubElement(
            root_cell,
            "mxCell",
            id=alloc_id(),
            value=label,
            style=style,
            vertex="1",
            parent=parent_id,
        )
        add_geometry(cell, x, y, width, height)
        return cell.attrib["id"]

    def add_class_cell(
        parent_id,
        cell_key,
        label,
        module_full,
        x,
        y,
        width=220,
        height=56,
        fill=None,
        stroke=None,
        include_module_name=True,
    ):
        value = label
        if include_module_name:
            value = f'{label}<br><font style="font-size:10px;">({module_full})</font>'
        style = "rounded=1;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;spacing=6;"
        if fill:
            style += f"fillColor={fill};"
        if stroke:
            style += f"strokeColor={stroke};"
        cell = ET.SubElement(
            root_cell,
            "mxCell",
            id=alloc_id(),
            value=value,
            style=style,
            vertex="1",
            parent=parent_id,
        )
        add_geometry(cell, x, y, width, height)
        node_ids[cell_key] = cell.attrib["id"]

    def add_edge(source, target, style):
        edge = ET.SubElement(
            root_cell,
            "mxCell",
            id=alloc_id(),
            value="",
            style=style,
            edge="1",
            parent="1",
            source=node_ids[source],
            target=node_ids[target],
        )
        ET.SubElement(edge, "mxGeometry", attrib={"relative": "1", "as": "geometry"})

    def layout_class_grid(parent_id, classes, x, y, columns=2):
        ranked_classes = collections.defaultdict(list)
        for cls in classes:
            ranked_classes[class_ranks.get(cls, 0)].append(cls)

        current_y = y
        for rank in sorted(ranked_classes):
            for index, cls in enumerate(sorted(ranked_classes[rank])):
                row = index // columns
                col = index % columns
                add_class_cell(
                    parent_id,
                    cls,
                    cls,
                    class_modules.get(cls, ""),
                    x + col * (DRAWIO_NODE_WIDTH + DRAWIO_NODE_X_GAP),
                    current_y + row * (DRAWIO_NODE_HEIGHT + DRAWIO_NODE_Y_GAP),
                    fill="#fff2cc" if cls in abc_classes else None,
                    stroke="#d79b00" if cls in abc_classes else None,
                    include_module_name=cls not in clustered_classes,
                )

            rank_rows = max(1, (len(ranked_classes[rank]) + columns - 1) // columns)
            current_y += rank_rows * (DRAWIO_NODE_HEIGHT + DRAWIO_NODE_Y_GAP) + 12

    def estimate_grid_height(classes, columns=2):
        ranked_classes = collections.defaultdict(list)
        for cls in classes:
            ranked_classes[class_ranks.get(cls, 0)].append(cls)

        total_height = 0
        for rank in sorted(ranked_classes):
            rank_rows = max(1, (len(ranked_classes[rank]) + columns - 1) // columns)
            total_height += rank_rows * 72 + 12
        return max(total_height - 12, 0)

    cluster_x = 20
    cluster_y = 20
    cluster_width = 520

    for cluster_module in cluster_modules:
        cluster_classes = clusters.get(cluster_module, [])
        if not cluster_classes:
            continue

        direct_classes, submodule_groups = group_cluster_classes_by_submodule(
            cluster_module, cluster_classes, class_modules
        )
        should_split = len(cluster_classes) > cluster_on_submodule_by and submodule_groups
        direct_height = estimate_grid_height(direct_classes) if direct_classes else 0
        nested_height = 0
        if should_split:
            for submodule_classes in submodule_groups.values():
                nested_height += 48 + estimate_grid_height(submodule_classes) + 12
        else:
            direct_height = estimate_grid_height(cluster_classes)

        cluster_height = max(120, 44 + direct_height + nested_height + 24)
        cluster_id = add_container("1", cluster_module, cluster_x, cluster_y, cluster_width, cluster_height)

        content_y = 36
        if should_split:
            if direct_classes:
                layout_class_grid(cluster_id, sorted(direct_classes), DRAWIO_CONTAINER_INSET_X, content_y)
                content_y += direct_height + 12
            for submodule_name, submodule_classes in sorted(submodule_groups.items()):
                sub_height = 48 + estimate_grid_height(submodule_classes)
                subcluster_id = add_container(
                    cluster_id,
                    submodule_name,
                    DRAWIO_CONTAINER_INSET_X,
                    content_y,
                    cluster_width - (DRAWIO_CONTAINER_INSET_X + 16),
                    sub_height,
                    dashed=True,
                )
                layout_class_grid(
                    subcluster_id,
                    sorted(submodule_classes),
                    DRAWIO_CONTAINER_INSET_X,
                    36,
                )
                content_y += sub_height + 12
        else:
            layout_class_grid(cluster_id, sorted(cluster_classes), DRAWIO_CONTAINER_INSET_X, content_y)

        cluster_y += cluster_height + 24

    if unclustered_classes:
        unclustered_height = 48 + estimate_grid_height(unclustered_classes)
        unclustered_id = add_container(
            "1",
            "Unclustered",
            cluster_x,
            cluster_y,
            cluster_width,
            unclustered_height,
        )
        layout_class_grid(unclustered_id, unclustered_classes, DRAWIO_CONTAINER_INSET_X, 36)

    external_base_keys = {}
    external_instantiation_keys = {}

    def ensure_external_base(parent_name):
        nonlocal external_y
        external_key = f"external:{parent_name}"
        if external_key in node_ids:
            return external_key
        add_class_cell(
            "1",
            external_key,
            parent_name,
            parent_name,
            cluster_x + cluster_width + 120,
            external_y,
            fill="#dae8fc",
            stroke="#6c8ebf",
            include_module_name=False,
        )
        external_base_keys[parent_name] = external_key
        external_y += 76
        return external_key

    def ensure_external_instantiation(class_name):
        nonlocal external_y
        external_key = f"external-inst:{class_name}"
        if external_key in node_ids:
            return external_key
        add_class_cell(
            "1",
            external_key,
            class_name,
            class_name,
            cluster_x + cluster_width + 120,
            external_y,
            fill="#dae8fc",
            stroke="#6c8ebf",
            include_module_name=False,
        )
        external_instantiation_keys[class_name] = external_key
        external_y += 76
        return external_key

    for cls, parents in class_hierarchy.items():
        if cls not in relevant_classes:
            continue
        for parent in parents:
            if parent != "ABC" and parent in relevant_classes and parent in node_ids:
                add_edge(
                    parent,
                    cls,
                    f"{DRAWIO_EDGE_STYLE}endArrow=block;strokeColor=#000000;",
                )

    for child, parent in external_inheritance:
        if child not in relevant_classes:
            continue
        if parent == "abc.ABC":
            continue
        if parent.split(".")[-1] not in RELEVANT_EXTERNAL_BASE_NAMES:
            continue
        external_key = ensure_external_base(parent)
        add_edge(
            external_key,
            child,
            f"{DRAWIO_EDGE_STYLE}endArrow=block;strokeColor=#4a86e8;",
        )

    for source, target in instantiations:
        if source not in relevant_classes or target not in relevant_classes:
            continue
        if (source, target) in memberships:
            continue
        add_edge(
            target,
            source,
            f"{DRAWIO_EDGE_STYLE}dashed=1;endArrow=open;strokeColor=#666666;",
        )

    for source, target in external_instantiations:
        if source not in relevant_classes:
            continue
        external_key = ensure_external_instantiation(target)
        add_edge(
            external_key,
            source,
            f"{DRAWIO_EDGE_STYLE}dashed=1;endArrow=open;strokeColor=#b85450;",
        )

    for source, target in memberships:
        if source not in relevant_classes or target not in relevant_classes:
            continue
        add_edge(
            target,
            source,
            f"{DRAWIO_EDGE_STYLE}dashed=1;endArrow=oval;strokeColor=#666666;",
        )

    tree = ET.ElementTree(root)
    output_path = f"{output_file}.drawio"
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def generate_class_diagram(
    class_hierarchy,
    class_modules,
    instantiations,
    external_inheritance,
    external_instantiations,
    references,
    memberships,
    output_file,
    abc_classes=None,
    cluster_modules=None,
    cluster_on_submodule_by=3,
    output_format="png",
    view=False,
):
    abc_classes = abc_classes or set()
    cluster_modules = cluster_modules or []
    dot = graphviz.Digraph(
        format=output_format,
        graph_attr={
            "rankdir": "BT",
            "newrank": "true",
            "ranksep": "1.0",
            "nodesep": "0.35",
        },
    )
    relevant_classes, clusters = get_diagram_scope(
        class_hierarchy, class_modules, external_inheritance, cluster_modules
    )

    clustered_classes = {cls for grouped_classes in clusters.values() for cls in grouped_classes}

    def add_class_node(graph, cls):
        is_abc = cls in abc_classes
        module_full = class_modules.get(cls, "")
        label = cls if cls in clustered_classes else f"{cls}\n({module_full})"

        if is_abc:
            graph.node(
                cls,
                label,
                style="filled",
                fillcolor="lightyellow",
                color="darkorange",
            )
        else:
            graph.node(cls, label)

    for cluster_module, cluster_classes in clusters.items():
        if not cluster_classes:
            continue
        with dot.subgraph(name=cluster_name_from_module(cluster_module)) as subgraph:
            subgraph.attr(label=cluster_module, style="rounded")
            subgraph.node(
                cluster_anchor_name(cluster_module),
                "",
                shape="point",
                width="0",
                height="0",
                style="invis",
            )
            if len(cluster_classes) > cluster_on_submodule_by:
                direct_classes, submodule_groups = group_cluster_classes_by_submodule(
                    cluster_module, cluster_classes, class_modules
                )
                for cls in direct_classes:
                    add_class_node(subgraph, cls)
                for submodule_name, submodule_classes in sorted(submodule_groups.items()):
                    with subgraph.subgraph(name=cluster_name_from_module(submodule_name)) as nested_subgraph:
                        nested_subgraph.attr(label=submodule_name, style="rounded,dashed")
                        for cls in submodule_classes:
                            add_class_node(nested_subgraph, cls)
            else:
                for cls in cluster_classes:
                    add_class_node(subgraph, cls)

    ordered_cluster_modules = [cluster_module for cluster_module in cluster_modules if clusters.get(cluster_module)]
    for first_module, second_module in zip(ordered_cluster_modules, ordered_cluster_modules[1:], strict=False):
        dot.edge(
            cluster_anchor_name(first_module),
            cluster_anchor_name(second_module),
            style="invis",
            weight="100",
        )

    for cls in sorted(relevant_classes - clustered_classes):
        add_class_node(dot, cls)

    for cls, parents in class_hierarchy.items():
        if cls not in relevant_classes:
            continue
        for parent in parents:
            if parent != "ABC" and parent in relevant_classes:
                dot.edge(parent, cls)

    for ext_inherit in external_inheritance:
        if ext_inherit[0] not in relevant_classes:
            continue
        # abc.ABC is conveyed via node colour, not as a separate node/edge.
        if ext_inherit[1] == "abc.ABC":
            continue
        if ext_inherit[1].split(".")[-1] not in RELEVANT_EXTERNAL_BASE_NAMES:
            continue
        dot.node(
            ext_inherit[1],
            ext_inherit[1],
            style="filled",
            fillcolor="lightblue",
            color="blue",
        )
        dot.edge(ext_inherit[1], ext_inherit[0], color="blue")

    for inst in instantiations:
        if inst[0] not in relevant_classes or inst[1] not in relevant_classes:
            continue
        # Skip if a membership edge already covers this relationship
        if (inst[0], inst[1]) in memberships:
            continue
        dot.edge(inst[1], inst[0], style="dotted")

    for ext_inst in external_instantiations:
        if ext_inst[0] not in relevant_classes:
            continue
        dot.node(
            ext_inst[1],
            ext_inst[1],
            style="filled",
            fillcolor="lightblue",
            color="blue",
        )
        dot.edge(ext_inst[1], ext_inst[0], style="dotted", color="blue")

    for mem in memberships:
        if mem[0] not in relevant_classes or mem[1] not in relevant_classes:
            continue
        dot.edge(mem[1], mem[0], style="dashed")

    # Add legend
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", style="dashed")
        legend.node("implements_a", "base class of", shape="plaintext")
        legend.node("implements_b", "", shape="plaintext")
        legend.node("inherits_external_a", "external base class of", shape="plaintext")
        legend.node("inherits_external_b", "", shape="plaintext")
        legend.node("member_of_a", "member of", shape="plaintext")
        legend.node("member_of_b", "", shape="plaintext")
        legend.node("instantiates_a", "instantiated by", shape="plaintext")
        legend.node("instantiates_b", "", shape="plaintext")
        legend.node("ext_instantiates_a", "externally instantiated by", shape="plaintext")
        legend.node("ext_instantiates_b", "", shape="plaintext")
        legend.node(
            "abc_example",
            "abstract base class",
            style="filled",
            fillcolor="lightyellow",
            color="darkorange",
        )
        legend.edge("implements_a", "implements_b", color="black")
        legend.edge("inherits_external_a", "inherits_external_b", color="blue")
        legend.edge("member_of_a", "member_of_b", style="dashed")
        legend.edge("instantiates_a", "instantiates_b", style="dotted")
        legend.edge("ext_instantiates_a", "ext_instantiates_b", style="dotted", color="blue")

    dot.render(output_file, view=view)


def main():
    parser = argparse.ArgumentParser(description="Generate a class hierarchy diagram from Python files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to Python files or directories to process.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="class_hierarchy",
        help="Output filename for the diagram (default: class_hierarchy).",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.",
    )
    parser.add_argument(
        "-m",
        "--base-module",
        default="mlcast",
        help="Base module to which class module names should be relative.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the rendered diagram in the default viewer after saving.",
    )
    parser.add_argument(
        "--cluster-by-module",
        nargs="+",
        default=[],
        help="Module prefixes used to group nodes into clusters.",
    )
    parser.add_argument(
        "--cluster-on-submodule-by",
        type=int,
        default=3,
        help=(
            "Split a requested module cluster into immediate submodule clusters "
            "when it contains more than this many classes."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "drawio"],
        default="png",
        help="Output format for the diagram.",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    (
        class_hierarchy,
        class_modules,
        instantiations,
        external_inheritance,
        external_instantiations,
        references,
        memberships,
        abc_classes,
    ) = extract_class_hierarchy_from_paths(args.paths, args.base_module)
    if not class_hierarchy:
        logger.info("No class definitions found in the specified paths.")
    else:
        if args.format == "drawio":
            output_path = write_drawio_diagram(
                class_hierarchy,
                class_modules,
                instantiations,
                external_inheritance,
                external_instantiations,
                memberships,
                args.output,
                abc_classes=abc_classes,
                cluster_modules=args.cluster_by_module,
                cluster_on_submodule_by=args.cluster_on_submodule_by,
            )
        else:
            generate_class_diagram(
                class_hierarchy,
                class_modules,
                instantiations,
                external_inheritance,
                external_instantiations,
                references,
                memberships,
                args.output,
                abc_classes=abc_classes,
                cluster_modules=args.cluster_by_module,
                cluster_on_submodule_by=args.cluster_on_submodule_by,
                output_format=args.format,
                view=args.open,
            )
            output_path = f"{args.output}.{args.format}"
        logger.info(f"Class hierarchy diagram saved as {output_path}")


if __name__ == "__main__":
    main()
