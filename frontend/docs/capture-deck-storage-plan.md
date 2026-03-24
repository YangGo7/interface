# Capture Deck Storage Plan

## Goal

Keep the current capture deck and empty-slot assignment feature lightweight for now, but leave a clear path to persistent storage later.

This note is for future implementation, not current behavior.

## Current Behavior

Current capture deck data is stored only in React state inside [ChartPage](/c:/interface/frontend/src/pages/ChartPage.tsx).

- `captureGallery`
- `assignedCaptureSlots`

Capture images are currently stored as in-memory `data:image/png;base64,...` strings.

This means:

- refresh clears the deck
- page revisit clears the deck
- another browser/device cannot see the deck
- nothing is written to cookie, localStorage, IndexedDB, or server DB yet

## Why Not Cookies

Cookies are not appropriate for capture images.

- size is too small
- every request would carry cookie payload
- binary/image payload is the wrong fit

Cookies should not be used for capture thumbnail or slot image persistence.

## localStorage

`localStorage` means browser-local persistence on the same machine and browser profile.

It is useful for:

- simple UI preferences
- last opened tab
- last selected layout
- lightweight mapping values

It is not a good long-term place for image payloads because:

- storage quota is small
- base64 images are large
- write/read becomes clumsy as capture count grows

## Recommended Persistence Layers

### Phase 1: Browser-only test persistence

Use:

- `localStorage` for metadata
- `IndexedDB` for image blobs

Suggested split:

- `localStorage`
  - sidebar open state
  - selected workspace tab
  - grid layout
  - slot-to-capture id mapping
  - last active case id

- `IndexedDB`
  - capture image blob
  - thumbnail blob
  - capture metadata if needed

This is the best fit for web testing without backend coupling.

### Phase 2: Server-backed persistence

When multi-user or multi-device continuity is needed, move persistence to backend storage.

Use:

- DB table/document for metadata
- object/file storage for image payload

Suggested server entities:

1. `case_session`
2. `capture_item`
3. `capture_slot_assignment`

## Proposed Data Model

### capture_item

- `id`
- `case_id`
- `created_at`
- `source_type`
  - `single_view`
  - `grid_viewport`
  - `image_view`
- `label`
- `width`
- `height`
- `mime_type`
- `blob_key` or `blob_path`
- `thumbnail_key` or `thumbnail_path`

### capture_slot_assignment

- `id`
- `case_id`
- `viewport_slot_id`
- `capture_item_id`
- `assigned_at`
- `sort_order`

### case_session

- `id`
- `job_id` or `report_session_id`
- `source_image_url`
- `viewer_mode`
- `grid_rows`
- `grid_cols`
- `created_at`
- `updated_at`

## Recommended Slot Strategy

Slot assignment should store only a reference, not duplicate image payload.

Recommended:

- slot stores `capture_item_id`
- actual image stays in one capture record

This avoids duplicate storage when the same capture is assigned to several slots.

## Browser Test Strategy

For browser-only persistence later:

1. keep `captureGallery` state as runtime source of truth
2. serialize lightweight capture metadata to `localStorage`
3. store capture image blobs in `IndexedDB`
4. restore on page load by joining metadata with blob records

## Migration Path

Planned order:

1. current in-memory state only
2. browser test persistence with `localStorage + IndexedDB`
3. backend API for capture metadata
4. backend blob/file storage
5. per-case shared persistence across devices

## Decision Summary

For future web testing:

- do not use cookies
- do not store full image payload only in `localStorage`
- use `localStorage` only for lightweight UI/session metadata
- use `IndexedDB` for browser-local capture image persistence
- move to DB + file storage when cross-device or shared persistence is required
