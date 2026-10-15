# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Iluvatar ``deepspeed`` operators.

Mirrors ``flag_train.deepspeed``: a specialised operator goes in the same-named
module here as the generic one it replaces -- ``deepspeed/blocked_flash.py``
beside ``flag_train/deepspeed/blocked_flash.py`` -- so which generic module an
override belongs to is readable from the path.

One operator so far. ``blocked_flash`` is here because the generic launcher
rejects this device outright, sizes its tile for more shared memory than the
BI-V150 has, and compiles its two-loop KV walk into wrong numbers here; the
module's docstring has all three. To specialise another, define it under this
package, re-export it here (``from .lamb import lamb``) and list it in
``__all__``, which is what ``SpecOpRegistrar`` registers.
"""

from .blocked_flash import blocked_flash

__all__ = ["blocked_flash"]
