"""Generate API reference tables from explicit exports and their source docstrings."""
import argparse
import ast
import importlib
import importlib.util
import inspect
import io
import re
import sys
import textwrap
import tokenize
from dataclasses import fields, is_dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs/reference/python-api.md"


@cache
def public_modules():
    """Return source modules that declare an explicit export list."""
    result = {}
    for path in sorted((ROOT / "avsectester").rglob("*.py")):
        tree = ast.parse(path.read_text())
        declaration = next((
            n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        ), None)
        if declaration is not None:
            name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            name = name.removesuffix(".__init__")
            result[name] = (path, tree, ast.literal_eval(declaration.value))
    return result


@cache
def architecture_rules():
    """Parse the architecture layer table and its type-only exceptions once."""
    text = (ROOT / "docs/concepts/architecture.md").read_text()
    rules = []
    exceptions = set()
    for line in text.splitlines():
        match = re.match(r"\|\s*L(\d)′?\s*\|", line)
        if match:
            for prefix in line.split("|")[2].strip().split(", "):
                prefix = prefix.strip("`").removesuffix("/").removesuffix(".py")
                name = "avsectester." + prefix.replace("/", ".")
                rules.append((name, int(match[1]) + (0.5 if "′" in match[0] else 0)))
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if (len(cells) == 3 and cells[0].startswith("avsectester.")
                and cells[1].startswith("avsectester.") and cells[2] == "TYPE_CHECKING"):
            exceptions.add((cells[0], cells[1]))
    return rules, exceptions


@cache
def layer(module):
    """Resolve a module against the shared architecture table."""
    if module == "avsectester":
        return 0
    rules, _ = architecture_rules()
    candidates = [(len(p), n) for p, n in rules
                  if module == p or module.startswith(p + ".")]
    if not candidates:
        raise ValueError(f"No architecture layer for {module}")
    return max(candidates)[1]


def paragraph(doc, key):
    """Require an object's own first documentation paragraph."""
    if not doc or not doc.strip():
        raise ValueError(f"Missing docstring: {key}")
    text = " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())
    return re.sub(r"``(.*?)``", r"`\1`", text)


@cache
def source_node(obj):
    """Locate a definition without falling back to inherited or built-in documentation."""
    obj = inspect.unwrap(obj)
    file = inspect.getsourcefile(obj)
    if file is None:
        raise ValueError(f"No Python source for {obj!r}")
    source = Path(file).read_text()
    tree = ast.parse(source)
    qualname = obj.__qualname__.split(".")
    body = tree.body
    node = None
    for name in qualname:
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        node = next((n for n in body if isinstance(n, kinds)
                     and n.name == name), None)
        if node is None:
            raise ValueError(f"No source definition: {obj.__module__}.{obj.__qualname__}")
        body = node.body
    return source, node


def fragment(source, node):
    """Preserve source expressions while making a single Markdown table cell.

    Line breaks collapse without bracket padding. Trailing commas before a closing bracket are
    dropped, except where a parenthesized single item would stop being a tuple.
    """
    segment = ast.get_source_segment(source, node)
    lines = segment.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    trailing, stack, previous = [], [], None
    for token in tokenize.generate_tokens(io.StringIO(segment).readline):
        if token.type in {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}:
            continue
        if token.string in {"(", "[", "{"}:
            stack.append([token.string, 0])
        elif token.string in {")", "]", "}"}:
            opener, commas = stack.pop()
            if previous is not None and previous.string == "," and (opener != "(" or commas > 1):
                trailing.append(offsets[previous.start[0] - 1] + previous.start[1])
        elif token.string == "," and stack:
            stack[-1][1] += 1
        previous = token
    for offset in reversed(trailing):
        segment = segment[:offset] + segment[offset + 1:]
    text = " ".join(segment.split())
    return re.sub(r" ([)\]}])", r"\1", re.sub(r"([(\[{]) ", r"\1", text))


def function_signature(obj, source, node, *, constructor=False, method=False):
    """Format original annotations/default expressions and verify runtime parameter names."""
    a = node.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + a.defaults
    names = [p.arg for p in positional]
    if a.vararg:
        names.append(a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append(a.kwarg.arg)
    actual = list(inspect.signature(obj).parameters)
    if actual != names:
        raise ValueError(f"AST/runtime parameter mismatch: {obj}: {names} != {actual}")

    def arg(p, default=None):
        text = p.arg
        if p.annotation is not None:
            text += ": " + fragment(source, p.annotation)
        if default is not None:
            text += (" = " if p.annotation is not None else "=") + fragment(source, default)
        return text
    hidden_required = [p.arg for p, default in [*zip(positional, defaults),
                                                *zip(a.kwonlyargs, a.kw_defaults)]
                       if constructor and p.arg.startswith("_") and default is None]
    if hidden_required:
        raise ValueError(f"Hidden required constructor parameters: {obj}: {hidden_required}")
    result = []
    for i, (p, default) in enumerate(zip(positional, defaults)):
        hidden = constructor and p.arg.startswith("_")
        receiver = (constructor or method) and i == 0 and p.arg in {"self", "cls"}
        if not (hidden or receiver):
            result.append(arg(p, default))
        if a.posonlyargs and i + 1 == len(a.posonlyargs):
            result.append("/")
    if a.vararg:
        result.append("*" + arg(a.vararg))
    elif a.kwonlyargs:
        result.append("*")
    result += [arg(p, default) for p, default in zip(a.kwonlyargs, a.kw_defaults)
               if not (constructor and p.arg.startswith("_"))]
    if a.kwarg:
        result.append("**" + arg(a.kwarg))
    text = "(" + ", ".join(result) + ")"
    if not constructor and node.returns is not None:
        text += " -> " + fragment(source, node.returns)
    return text


def class_signature(cls, source, node):
    """Describe explicit constructors, dataclass fields, exceptions and protocols."""
    if getattr(cls, "_is_protocol", False):
        return ""
    if issubclass(cls, BaseException):
        return ""
    init = next((n for n in node.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is not None:
        return function_signature(cls.__init__, source, init, constructor=True)
    if is_dataclass(cls):
        definitions = {}
        for base in reversed(cls.__mro__):
            if base.__module__ == "builtins":
                continue
            try:
                src, definition = source_node(base)
            except (TypeError, ValueError):
                continue
            for field in definition.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                    definitions[field.target.id] = (src, field)
        params = inspect.signature(cls).parameters
        if set(params) != {f.name for f in fields(cls) if f.init}:
            raise ValueError(f"Dataclass constructor mismatch: {cls}")
        result = []
        keyword = False
        for name, parameter in params.items():
            if name.startswith("_"):
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"Hidden required dataclass field: {cls}: {name}")
                continue
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword:
                result.append("*")
                keyword = True
            src, field = definitions[name]
            text = name + ": " + fragment(src, field.annotation)
            value = field.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "field"):
                default = next((k for k in value.keywords
                                if k.arg in ("default", "default_factory")), None)
                if default is not None:
                    value = default.value
                    if default.arg == "default_factory":
                        if isinstance(value, ast.Lambda):
                            rendered = fragment(src, value.body)
                        elif isinstance(value, (ast.Name, ast.Attribute)):
                            rendered = fragment(src, value) + "()"
                        else:
                            rendered = "<factory>"
                    else:
                        rendered = fragment(src, value)
                    text += " = " + rendered
            elif value is not None:
                text += " = " + fragment(src, value)
            result.append(text)
        return "(" + ", ".join(result) + ")"
    for base in cls.__mro__[1:]:
        if base.__module__ == "builtins":
            continue
        init = base.__dict__.get("__init__")
        if init is not None:
            inherited_source, inherited_node = source_node(init)
            return function_signature(init, inherited_source, inherited_node, constructor=True)
    return "()"


def assignment_docs(body):
    """Read attribute descriptions immediately after their assignment statements."""
    result = {}
    for i, n in enumerate(body[:-1]):
        following = body[i + 1]
        if not (isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)):
            continue
        targets = (n.targets if isinstance(n, ast.Assign) else
                   [n.target] if isinstance(n, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name):
                result[target.id] = following.value.value
    return result


def is_public_class(cls):
    """Identify classes listed by an owning public module."""
    data = public_modules().get(cls.__module__)
    return data is not None and cls.__name__ in data[2]


def public_bases(cls):
    """Render bases that are public extension interfaces or external types."""
    bases = [
        base.__name__ for base in cls.__bases__
        if base.__name__ not in {"object", "ABC", "Protocol"} and not base.__name__.startswith("_")
    ]
    return "Subclass of " + ", ".join(f"`{base}`" for base in bases) + ". " if bases else ""


def callable_meaning(value, node, label):
    """Require a callable contract and identify abstract/context-manager behavior."""
    meaning = paragraph(ast.get_docstring(node, clean=False), label)
    context_manager = any(
        isinstance(decorator, ast.Name) and decorator.id == "contextmanager"
        or isinstance(decorator, ast.Attribute) and decorator.attr == "contextmanager"
        for decorator in node.decorator_list)
    if (context_manager and node.returns is not None
            and "Iterator" in ast.unparse(node.returns)):
        meaning = "Context manager. " + meaning
    if getattr(value, "__isabstractmethod__", False):
        meaning = "Abstract. " + meaning
    return meaning


def attribute_description(cls, name):
    """Find the nearest explicit description of a class attribute."""
    for base in cls.__mro__:
        if base.__module__ == "builtins":
            continue
        _, node = source_node(base)
        description = assignment_docs(node.body).get(name)
        if not description:
            member = next((item for item in node.body
                           if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and item.name == name), None)
            description = ast.get_docstring(member) if member is not None else None
        if description:
            return description
    return None


def member_rows(cls, name, module):
    """Render owned members and public members inherited from private bases."""
    field_names = {f.name for f in fields(cls)} if is_dataclass(cls) else set()
    seen = set()
    rows = []
    for base in cls.__mro__:
        if base.__module__ == "builtins" or base is not cls and is_public_class(base):
            continue
        if not base.__module__.startswith("avsectester"):
            continue
        if base is not cls and any(
            is_public_class(parent) and base in parent.__mro__
            for parent in cls.__mro__[1:]
        ):
            continue
        source, node = source_node(base)
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = member.name
                if key in seen:
                    continue
                value = base.__dict__.get(key)
                abstract = getattr(value, "__isabstractmethod__", False)
                abstract_dunder = abstract and key.startswith("__") and key.endswith("__")
                if key.startswith("_") and key != "__call__" and not abstract_dunder:
                    continue
                if any(isinstance(d, ast.Attribute) and d.attr in {"setter", "deleter"}
                       for d in member.decorator_list):
                    continue
                seen.add(key)
                inherited = any(key in parent.__dict__ for parent in cls.__mro__[1:])
                meaning_node = member
                if (inherited and not ast.get_docstring(member)
                        and key != "__call__" and not abstract):
                    parent = next(parent for parent in cls.__mro__[1:]
                                  if key in parent.__dict__)
                    if any(is_public_class(base) and parent in base.__mro__
                           for base in cls.__mro__[1:]):
                        continue
                    _, parent_node = source_node(parent)
                    meaning_node = next(n for n in parent_node.body
                                        if isinstance(n, ast.FunctionDef) and n.name == key)
                label = module + "." + name + "." + key
                if isinstance(value, property):
                    function = value.fget
                elif isinstance(value, (classmethod, staticmethod)):
                    function = value.__func__
                else:
                    function = value
                meaning = callable_meaning(function, meaning_node, label)
                if isinstance(value, property):
                    if member.returns is None:
                        raise ValueError(f"Missing property return annotation: {label}")
                    signature = ": " + fragment(source, member.returns)
                else:
                    signature = function_signature(
                        function, source, member, method=not isinstance(value, staticmethod))
                rows.append((name + "." + key + signature, meaning))
            elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                targets = member.targets if isinstance(member, ast.Assign) else [member.target]
                for target in targets:
                    if (not isinstance(target, ast.Name) or target.id.startswith("_")
                            or target.id in seen or target.id in field_names):
                        continue
                    key = target.id
                    seen.add(key)
                    description = paragraph(
                        attribute_description(cls, key), module + "." + name + "." + key)
                    annotation = (": " + fragment(source, member.annotation)
                                  if isinstance(member, ast.AnnAssign) else "")
                    default = " = " + fragment(source, member.value) if member.value else ""
                    rows.append((name + "." + key + annotation + default, description))
    return rows


def imported_names(module, path, tree):
    """Resolve explicit source imports for re-export links."""
    imported = {}
    for item in tree.body:
        if not isinstance(item, ast.ImportFrom):
            continue
        owner = item.module or ""
        if item.level:
            package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
            owner = importlib.util.resolve_name("." * item.level + owner, package)
        for alias in item.names:
            imported[alias.asname or alias.name] = owner
    return imported


def rows_for(module, data):
    """Combine explicit definitions, constants and linked re-exports."""
    path, tree, exports = data
    aliases = {
        target.id: node.value
        for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    loaded = importlib.import_module(module)
    source_text = path.read_text()
    imported = imported_names(module, path, tree)
    assignments = assignment_docs(tree.body)
    definitions = {node.name for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    rows = []
    for name in exports:
        obj = getattr(loaded, name)
        owner = imported.get(name)
        if owner is None and name not in definitions and name not in aliases:
            owner = getattr(obj, "__module__", None)
        owner = owner or module
        if name not in definitions and owner != module and owner in public_modules():
            rows.append((name, f"[{name}](#" + owner.replace(".", "") + ")"))
        elif inspect.isclass(obj):
            source, node = source_node(obj)
            signature = class_signature(obj, source, node)
            meaning = public_bases(obj) + paragraph(ast.get_docstring(node), module + "." + name)
            if "__enter__" in obj.__dict__:
                meaning = "Context manager. " + meaning
            rows.append((name + signature, meaning))
            rows.extend(member_rows(obj, name, module))
        elif inspect.isfunction(obj):
            source, node = source_node(obj)
            rows.append((name + function_signature(obj, source, node),
                         callable_meaning(obj, node, module + "." + name)))
        else:
            expression = aliases.get(name)
            signature = (name + " = " + fragment(source_text, expression)
                         if expression is not None else name)
            rows.append((signature, paragraph(assignments.get(name), module + "." + name)))
    return rows


def module_block(module, data):
    """Render a deterministic generated module block."""
    doc = paragraph(ast.get_docstring(data[1], clean=False), module)
    purpose = re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0]
    purpose_lines = textwrap.wrap(purpose, width=96, break_long_words=False, break_on_hyphens=False)
    if len(purpose_lines) > 1 and len(purpose_lines[-1].split()) == 1:
        previous, word = purpose_lines[-2].rsplit(" ", 1)
        purpose_lines[-2:] = [previous, word + " " + purpose_lines[-1]]
    lines = [f"<!-- api:begin module={module} -->", "\n".join(purpose_lines), ""]
    if not data[2]:
        lines += ["No exports; import from submodules."]
    else:
        lines += ["| Name and signature | Meaning |", "|---|---|"]
        for name, meaning in rows_for(module, data):
            signature = name.replace("|", r"\|")
            description = meaning.replace("|", r"\|")
            lines.append(f"| `{signature}` | {description} |")
    return "\n".join(lines) + "\n<!-- api:end -->"


def render(text):
    """Regenerate module ordering and tables while retaining manual constraints and links."""
    modules = public_modules()
    ordered = sorted(modules, key=lambda m: (layer(m), m))
    sections = {}
    matches = list(re.finditer(r"^### (.+)$", text, re.M))
    first = matches[0].start() if matches else len(text)
    before = text[:first]
    structure = re.fullmatch(
        r"(?P<prefix># Python API reference\n(?:\n|>[^\n]*\n)*)"
        r"## Modules\s*<!-- api:begin module=module-map -->.*?<!-- api:end -->"
        r"\s*## Public modules\s*", before, re.S)
    if structure is None:
        raise ValueError("Unrecognized handwritten content before the first module")
    prefix = structure["prefix"].rstrip()
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[match.end():end]
        manual = re.sub(r"<!-- api:begin.*?<!-- api:end -->", "", section, flags=re.S).strip()
        allowed = {"Constraints and errors", "Extension contract", "See also"}
        headings = list(re.finditer(r"^(#{1,6}) (.+)$", manual, re.M))
        if match[1] not in modules:
            raise ValueError(f"Non-public module section: {match[1]}")
        if match[1] in sections:
            raise ValueError(f"Duplicate module section: {match[1]}")
        if manual and (not headings or manual[:headings[0].start()].strip()
                       or any(heading[1] != "####" or heading[2] not in allowed
                              for heading in headings)):
            raise ValueError(f"Unrecognized handwritten content in {match[1]}")
        sections[match[1]] = manual
    diagram = ["<!-- api:begin module=module-map -->", "| Layer | Public module |", "|---|---|"]
    for module in ordered:
        rank = layer(module)
        label = str(int(rank)) + ("′" if rank % 1 else "")
        diagram.append(f"| L{label} | [`{module}`](#{module.replace('.', '')}) |")
    diagram += ["<!-- api:end -->"]
    output = [prefix, "## Modules", "\n".join(diagram), "## Public modules"]
    for module in ordered:
        output += ["### " + module, module_block(module, modules[module])]
        manual = sections.get(module, "")
        if manual:
            output.append(manual)
    return "\n\n".join(output).rstrip() + "\n"


def main():
    """Write the reference or verify that its generated blocks are current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the saved reference differs.")
    args = parser.parse_args()
    try:
        current = TARGET.read_text()
        generated = render(current)
    except (ValueError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    if args.check:
        if current != generated:
            parser.exit(1, "API reference differs; run write_api_reference.py.\n")
    else:
        TARGET.write_text(generated)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
