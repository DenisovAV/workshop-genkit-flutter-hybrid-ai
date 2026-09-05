# Architecture-review fixes for the hybrid-AI workshop app

Spec: the architectural review of 2026-09-05 (three passes: Flutter architect, Genkit
architect, and a step-back review), with two findings reproduced on a device:
- On-device model + RAG overflow: `Input token ids are too long … 1713 >= 1024` when
  RagService retrieves 3 city guides (each ~540-650 tokens); the plugin defaults
  `maxTokens` (the WHOLE context window, input + output) to 1024; the `.litertlm` file
  is built for 4096 (`ekv4096`).
- `WithFallback` around Smart adds nothing for text (CapabilityStrategy already returns
  `[kCloud, kOnDevice]`) and appends a blind text-only branch for images.

## Global Constraints

- This is a WORKSHOP teaching codebase. Fewer lines that teach the same thing win.
  NO new packages, NO state-management library, NO DI, NO repository layer.
- Do not touch `android/`, `ios/`, `macos/`, `pubspec.yaml`, `assets/`.
- `flutter analyze` must report 0 issues and `flutter test` must be green after every task.
- Every task ends in a commit on this branch. Commit with
  `--author="Sasha Denisov <denisov.shureg@gmail.com>"` and end the message body with
  the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Keep the existing shape: `AiEngine` owns the Genkit instance and policy composites;
  `RagService` owns retrieval; `ChatScreen` is plain `setState` UI.
- Branch keys are the existing `kOnDevice` / `kCloud` constants from `genkit_hybrid`.
- `strategyFor(PolicyMode.cascade)` keeps throwing `ArgumentError` (documented in the
  published codelab; changing it ripples into a live document).

### Task 1: Smart — drop the `WithFallback` wrapper

File: `lib/services/ai_engine.dart` (`strategyFor`, `case PolicyMode.smart`),
`test/ai_engine_policy_test.dart`.

Replace the smart case body with a bare `CapabilityStrategy`:

```dart
      case PolicyMode.smart:
        // Image → cloud only (only it declares vision). Text → both qualify,
        // cloud-first in `supports` insertion order, on-device as the tail.
        // No WithFallback: CapabilityStrategy already yields the on-device
        // tail for text, and for an image a forced on-device tail would hand
        // the picture to a model that cannot see it. Offline + image should
        // fail loudly, not silently degrade to text-only.
        return CapabilityStrategy(
          supports: {
            kCloud: {ModelCapability.vision},
            kOnDevice: <ModelCapability>{},
          },
        );
```

Remove the now-unused `WithFallback` usage (keep the import only if something else
uses it). Tests:
- `smart mode: text is cloud-first with on-device fallback` keeps asserting
  `[kCloud, kOnDevice]`.
- The image test must assert the exact route `[kCloud]` (not `route.first`), and its
  NOTE comment about "no observable way to assert" is deleted — it is now observable.

### Task 2: Context budget for the on-device branch

File: `lib/services/ai_engine.dart`, `test/ai_engine_policy_test.dart`.

Add, next to the other top-level constants:

```dart
/// Context window for the on-device branch, in tokens. `maxTokens` is the
/// WHOLE window (input + output) and genkit_flutter_gemma defaults it to 1024.
/// RagService's take(3) of ~600-token city guides alone is ~1.7k tokens —
/// measured on device: "Input token ids are too long … 1713 >= 1024". The
/// bundled Gemma-3-1B `.litertlm` is built for 4096 (`ekv4096`), so use it.
const kOnDeviceContextTokens = 4096;
```

Wrap the resolved on-device `Model` so every request reaching it carries
`config['maxTokens'] = kOnDeviceContextTokens` unless the request already sets
`maxTokens`. The cloud branch must NOT receive `maxTokens`. Implement as a private
`Model _withContextBudget(Model inner)` that builds a `Model(name: '${inner.name}/ctx',
fn: (request, context) { …merge config…; return inner.fn(request, context); })` —
forwarding the same `context` keeps streaming intact (genkit_hybrid's `runInOrder`
calls branches exactly as `branch.fn(request, context)`). Merge into a fresh map; if
`ModelRequest` exposes no `config` setter, build a copy rather than mutating a shared
object. Apply the wrapper in BOTH places that set `_local`: `initialize()` (after
`_resolve`) and `AiEngine.forTest` (so tests see the same wiring).

Tests (use `AiEngine.forTest` + fake branches that capture `request.config`):
- on-device branch via `modelFor(PolicyMode.local)` receives `maxTokens == 4096`;
- cloud branch via `modelFor(PolicyMode.cloud)` receives no `maxTokens` key;
- a request whose config already has `maxTokens: 2048` reaches on-device with 2048.

### Task 3: `PolicyMode` carries its facts; one switch remains

Files: `lib/services/ai_engine.dart`, `lib/screens/chat_screen.dart`,
`test/ai_engine_policy_test.dart`.

Replace the bare enum with an enhanced enum. Labels are the EXACT strings the dropdown
shows today:

```dart
enum PolicyMode {
  cloud('Cloud', needsLocal: false),
  local('Local', needsCloud: false, textOnly: true),
  smart('Smart (image-aware)'),
  cascade('Cascade (escalate on quality)', textOnly: true),
  budget('Budget (cost-gated)');

  const PolicyMode(
    this.label, {
    this.needsCloud = true,
    this.needsLocal = true,
    this.textOnly = false,
  });

  /// Dropdown text.
  final String label;
  /// Which branches the composite for this mode requires.
  final bool needsCloud;
  final bool needsLocal;
  /// True when the primary route starts on the text-only on-device model, so
  /// an attached image cannot be handled (the UI blocks send with a hint).
  final bool textOnly;

  bool availableWith({required bool cloud, required bool local}) =>
      (!needsCloud || cloud) && (!needsLocal || local);
}
```

Then:
- Delete `_hasRequiredBranches` and `requiresTextOnly`. `_registerPolicyModels` uses
  `mode.availableWith(cloud: _cloud != null, local: _local != null)`.
- `strategyFor` and `_buildModel` stay as they are (they are the lesson: mode →
  genkit_hybrid construct).
- `chat_screen.dart`: build the dropdown items from `PolicyMode.values` with
  `Text(mode.label)` and `enabled: mode.availableWith(cloud: _engine.cloudReady,
  local: _engine.localReady)`; delete the `_cloudReady`/`_localReady` fields and read
  `_engine.cloudReady`/`_engine.localReady` at every former use (the `defaultPolicy`
  switch, the `parts` list, the dropdown). Replace `_engine.requiresTextOnly(_policy)`
  with `_policy.textOnly`.
- Tests: replace the `requiresTextOnly` test with a table test over
  `PolicyMode.values` asserting `label`, `textOnly`, and `availableWith` for the four
  (cloud, local) combinations. In the cloud-absent and local-absent guard groups,
  collapse the repeated `expect(() => engine.modelFor(x), throwsA(isA<StateError>()))`
  blocks into a loop over `PolicyMode.values.where((m) => !m.availableWith(...))`.

### Task 4: `AiEngine.send()` owns the turn; the widget shrinks

Files: `lib/services/ai_engine.dart`, `lib/screens/chat_screen.dart`,
`test/ai_engine_policy_test.dart`.

Add to `AiEngine`:

```dart
  /// One chat turn: builds the user message (text, plus an image as a data
  /// URI with an explicit image/* contentType — the on-device plugin drops
  /// media without one and CapabilityStrategy reads it to detect vision),
  /// streams the reply through the composite for [mode], and does the budget
  /// accounting afterwards. [prompt] is the final prompt — already
  /// RAG-augmented by the caller when RAG is on.
  Stream<String> send(
    PolicyMode mode,
    String prompt, {
    Uint8List? imageBytes,
    String? imageMime,
  }) async* { … }
```

Move into it, verbatim in behaviour, from `_sendMessage`: the `content`/`MediaPart`/
base64 construction, the `wasBudgetAvailable` snapshot, the `generateStream` call
(yielding `chunk.text` only when non-empty is NOT required — yield every chunk's text
as today), and the post-stream `cloudCallsSpent` accounting including its existing
"best-effort demo counter" comment. Accounting runs only after the stream completes
without throwing (same as today).

`_sendMessage` then: keeps the `_policy.textOnly` image guard, the RAG augmentation,
the placeholder message, the 50 ms UI throttle loop over `_engine.send(...)`, and the
`(no response)` / `Error:` handling. Delete the `wasBudgetAvailable` local, the
`content`/`userMessage` construction, the `cloudCallsSpent` block, and the imports that
become unused (`dart:convert`, `package:genkit/genkit.dart` if nothing else uses them).

Send gate: the send `IconButton.filled`'s `onPressed` must also be `null` when
`!(_engine.cloudReady || _engine.localReady)` (status already reads "No services
available" in that state).

The placeholder-removal branch in `finally` (`if (!generationSucceeded && … .text.isEmpty)
_messages.removeLast()`): every path either sets `generationSucceeded`, writes a
non-empty `Error: …` message, or returns on `!mounted` (making `finally`'s own
`mounted` check false). Confirm that by reading the paths, state the confirmation in
the report, and delete the branch. If you find a path that reaches it, keep it and
say which path.

Tests (`AiEngine.forTest` with fake branches, counting calls):
- `send(cloud, …)` → `cloudCallsSpent` increments by 1;
- `send(budget, …)` with budget available → increments by 1;
- `send(budget, …)` after `cloudCallsSpent = budgetCap` → does not increment;
- `send(local, …)` → does not increment;
- `send(smart, 'hi', imageBytes: [0], imageMime: 'image/png')` → the message that
  reaches the branch has a `MediaPart` whose `contentType == 'image/png'`.

### Task 5: RagService and dispose cleanup

Files: `lib/services/rag_service.dart`, `lib/services/ai_engine.dart`,
`lib/screens/chat_screen.dart`, `integration_test/smoke_test.dart`.

- `_cosine`: replace the `assert` with `if (a.length != b.length) throw ArgumentError(
  'Embedding dimensions must match: ${a.length} != ${b.length}');` — an `assert` is
  stripped from release builds and the loop would then throw a RangeError.
- Name the magic numbers: `const kMinSimilarity = 0.5;` and `const kTopK = 3;` at top
  level with one-line comments, used at the `where` and `take` sites.
- Delete public getters that nothing outside the class reads (`isInitialized`,
  `documentCount`) — grep `lib/` and `test/` and `integration_test/` first; keep any
  with a reader and say so in the report.
- `RagService.dispose()` and `AiEngine.dispose()` contain no `await`: change both
  signatures to `void dispose()`. Update the two call sites in `chat_screen.dart`'s
  `dispose()` (they are already called without `await`).
- `integration_test/smoke_test.dart`: remove the `if (engine.localReady)` guard that
  follows `expect(engine.localReady, isTrue)` — it is dead.
- Unit tests must stay green; no new tests required beyond keeping `flutter test` green.
