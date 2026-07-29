# OrcaHand Description In DexSlide

This directory vendors the local `orcahand_description` assets needed by DexSlide retargeting.

## Included assets

- `models/urdf/orcahand_right.urdf`
- `models/urdf/orcahand_left.urdf`
- `models/urdf/orcahand_right_extended.urdf`
- `models/urdf/orcahand_left_extended.urdf`
- matching `assets/left/` and `assets/right/` meshes
- `retargeting/orcahand_v2_right_vector_semantic_12d.json` as the default DexSlide live vector-retarget config
- `retargeting/orcahand_v1_right_vector_21d.json` kept as a legacy reference config

## Integrity check

The copied URDF set was checked before vendoring.

- `orcahand_right.urdf`: 62 / 62 mesh references resolved
- `orcahand_left.urdf`: 62 / 62 mesh references resolved
- `orcahand_right_extended.urdf`: 68 / 68 mesh references resolved
- `orcahand_left_extended.urdf`: 68 / 68 mesh references resolved

## Notes for retargeting

- DexSlide uses `assets/skeletons/skeleton.json` as the human-side kinematic source.
- The live wrapper does not require a human URDF.
- The default DexSlide bridge fixes `right_wrist = 0` because the glove stream currently carries 20 finger joints, not a wrist joint.
- `orcahand_v2_right_vector_semantic_12d.json` uses only dynamic, controllable link vectors and can warm-start from the direct joint map.
- If the glove layout changes, update `human_joint_names` and the human landmark indices in the active retarget JSON.
- If OrcaHand link naming changes, update `orcahand_urdf_joint_names`, `target_origin_link_names`, and `target_task_link_names` together.
