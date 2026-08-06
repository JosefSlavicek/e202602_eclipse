"""Read cached v2 pickles under the v5 package name.

A v2 run's `v2-stage0/1/2.pkl` name their classes as `eclipse_v2.stage0.ImageInfo`, so
`pickle.load` re-imports `eclipse_v2`.  v5 is a vendored copy — the classes are identical
in layout — so aliasing the module names is sufficient and no unpickler subclass is needed.

Reusing a cached run is the difference between a 50-minute registration and none, which is
why this exists at all.  It is opt-in: nothing imports it implicitly, because silently
answering to another version's package name is exactly the kind of cross-version coupling
the vN convention exists to prevent.

Usage:
    from eclipse_v5.compat import install_v2_pickle_aliases
    install_v2_pickle_aliases()
    ...   # now `pickle.load` on v2-stage1.pkl works
"""
from __future__ import annotations

import sys

# Every submodule a v2 pickle can name. stage0 holds ImageInfo (the only class actually
# pickled today); the rest are listed so an older or newer pickle referencing them resolves
# rather than dying halfway through a load.
_ALIASED_SUBMODULES = (
    "coords", "device", "display", "inputs", "stage0", "stage1", "stage2", "stage3", "utils",
)


def install_v2_pickle_aliases() -> list[str]:
    """Alias `eclipse_v2[.sub]` -> `eclipse_v5[.sub]` in `sys.modules`. Returns names installed.

    Uses `setdefault` semantics: a real `eclipse_v2` already imported into this interpreter
    wins, so a process that legitimately holds both versions is never hijacked.
    """
    import eclipse_v5

    installed = []
    if "eclipse_v2" not in sys.modules:
        sys.modules["eclipse_v2"] = eclipse_v5
        installed.append("eclipse_v2")
    for name in _ALIASED_SUBMODULES:
        alias = f"eclipse_v2.{name}"
        if alias in sys.modules:
            continue
        module = __import__(f"eclipse_v5.{name}", fromlist=[name])
        sys.modules[alias] = module
        installed.append(alias)
    return installed
