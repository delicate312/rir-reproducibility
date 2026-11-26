"""Verify a source checkout without importing executable training runners."""
from pathlib import Path
import ast, json, hashlib
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for rel,expected in manifest['files'].items():
 p=root/rel
 assert p.is_file() and not p.is_symlink(), rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, rel
 if p.suffix=='.py': ast.parse(p.read_text(), filename=rel)
assert (root/'02_alt_temporal_train.py').is_file()
assert (root/'relation_schema.py').is_file()
print('Source checksums, syntax and dynamic training dependencies verified.')
