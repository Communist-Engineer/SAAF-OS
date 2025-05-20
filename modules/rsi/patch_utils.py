from typing import List
from modules.rsi.engine import Patch
import copy

def generate_patch_family(seed_patch: Patch, pattern: str) -> List[Patch]:
    """
    Generate a family of patches from a seed_patch by applying a pattern.
    For example, generalize a planner parameter patch to multiple modules.
    Args:
        seed_patch: The original Patch object
        pattern: A string pattern, e.g., 'planner', 'module_*', etc.
    Returns:
        List of Patch objects
    """
    patch_family = []
    # Example: if pattern is a list of module names, generate a patch for each
    if pattern.startswith('module:'):
        modules = pattern.split(':', 1)[1].split(',')
        for mod in modules:
            new_patch = copy.deepcopy(seed_patch)
            new_patch.module_path = mod.strip()
            new_patch.patch_id = f"{seed_patch.patch_id}_to_{mod.strip()}"
            patch_family.append(new_patch)
    else:
        # Default: just return the seed_patch in a list
        patch_family = [seed_patch]
    return patch_family

# Example test case
if __name__ == '__main__':
    dummy = Patch(module_path='modules/planner.py', content='...', description='Test', patch_id='seed')
    fam = generate_patch_family(dummy, 'module:modules/planner.py,modules/fwm.py')
    for p in fam:
        print(p.patch_id, p.module_path)
