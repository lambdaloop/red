# Tailcycle export: 2D-only and multi-video datasets

## Goal

Make Tailcycle export represent a genuinely 2D annotation dataset without inventing camera calibration, and export projects with multiple videos as multiple Tailcycle sessions (one session per video). Keep the resulting dataset importable through the existing multi-session import workflow.

## Current behavior and constraints

- `src/tailcycle_export.cpp` currently requires one calibration per camera, validates 3D camera properties and image dimensions, and always writes `calibration.toml`. It writes `mode = "2d"` for fewer than two cameras, but that does not make the export calibration-free.
- `src/tailcycle_import.cpp` currently rejects a session without `calibration.toml`, including `mode = "2d"` sessions. It derives the camera list and image sizes from that file; image height is also needed to convert between Tailcycle's top-left and red's bottom-left 2D coordinates.
- `export_tailcycle()` in `src/export_formats.h` currently builds one session from the loaded media folder and all configured cameras. It extracts one media stream per camera into that session's group.
- `TailcycleImport::Session` represents one selected group, and the existing import UI/workflow should be preserved rather than replaced.

## Proposed behavior

1. **Minimal 2D calibration metadata (per Tailcycle spec)**
   - `calibration.toml` is required for every session, including `mode = "2d"` (§5 and validation rule 4). It is also the only place the camera name and image size live; the name must match the media directory/video under every group.
   - For a 2D-only camera, write the required `name`, `size`, and `offset = [0, 0]`. The spec explicitly permits omitting `matrix`, `rotation`, `translation`, and `distortions`; do not invent real calibration values.
   - Read image dimensions from the video and write them to `size`. Keep the importer reading camera name and size from `calibration.toml`, which it already does; adjust its validation/parsing only if necessary to accept the permitted omitted geometry fields for 2D.
   - Continue requiring full calibration for 3D sessions. Preserve existing calibrated 2D behavior.

2. **One session per video**
   - In 2D mode, automatically create a separate session for each source video, rather than treating unrelated videos as synchronized cameras in one session.
   - Use a meta dataset root with the Tailcycle layout `<dataset>/<split>/<session>/` (for example `<dataset>/train/<video-stem>/`). This is a shared dataset directory containing multiple session folders under each split, not one output dataset per video.
   - Each session gets its own camera identity, group and extracted media, source range/frame count, and relevant per-video annotations. Use the video stem as the session ID; sanitize it and detect duplicate stems before writing anything, rather than silently colliding.
   - Preserve the existing label-source split behavior (`annotated`/`tracked`) consistently for each video's session, and avoid partial/colliding output when one video fails.
   - Keep the current single-session export path for actual multi-camera/synchronized 3D projects.

## Implementation sequence

1. **Confirm format contract**: Check the Tailcycle dataset schema/consumer requirements for calibration-free `mode = "2d"` sessions and where camera names and image dimensions belong. Decide the minimal valid metadata representation before editing the writer/reader.
2. **Model 2D metadata**: Extend the export/import config and session representation only as needed to carry camera names and image dimensions without calibration. Keep 3D validation and serialization unchanged.
3. **Implement calibration-free 2D export/import**: Branch on actual export mode/layers; make the reader accept calibration omission only for 2D, and preserve support for calibrated 2D sessions. Add focused round-trip and invalid-input tests.
4. **Implement per-video session export**: Identify the existing project's source-video enumeration and frame/annotation indexing. Add an orchestrator that partitions annotations by video and invokes the session writer with stable session/group IDs and corresponding media extraction. Preserve current one-session behavior for synchronized camera sets.
5. **UI and compatibility**: Expose the new choice/path in the export UI without changing existing defaults unexpectedly. Ensure the generated multi-session dataset appears as multiple sessions to the existing importer and that selecting/importing sessions retains expected behavior.
6. **Verify**: Run Tailcycle export/import tests and relevant project tests; test 2D without calibration, 2D with legacy calibration, 3D with calibration, multiple independent videos, name collisions, and export failures.

## Acceptance criteria

- A purely 2D dataset exports and imports without any calibration file or fabricated calibration values.
- A calibration-free 2D session retains correct camera identity, dimensions, frame indices, and point coordinates through round-trip.
- A multi-video project can export one independently importable session per video, each containing only its own frames/annotations/media.
- Existing 3D export/import, calibrated 2D datasets, and synchronized multi-camera sessions continue to work.
- Failures (unsupported metadata, duplicate session IDs, missing video, inconsistent dimensions) are reported before or without leaving misleading partial sessions.

## Decisions captured

- Every session, including 2D, has `calibration.toml`. For 2D, write the required camera name, size and zero offset; omit unneeded matrix/extrinsics/distortion instead of inventing 3D calibration.
- Automatically export 2D videos as separate sessions. Keep synchronized multi-camera export for 3D.
- Read image dimensions directly from video media and write them to the camera's `size` in `calibration.toml`.
- Use the video stem as each session name (sanitized; reject duplicate stems before writing).
- Export into one meta dataset tree, `<dataset>/<split>/<session>/`, such as `<dataset>/train/video01/`.

## Format contract reviewed

`~/research/janelia/tailcycle/tailcyclenet/docs/annotation_format.md` explicitly requires `calibration.toml` in every session (§5, validation rule 4). A 2D camera may omit `matrix`, `rotation`, `translation`, and `distortions`, but must declare `name`, `size`, and `offset`; the spec says to use `offset = [0, 0]` for 2D. No open format-contract question remains.
