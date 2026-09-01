# Workshop 2 → genkit_hybrid migration — Design

**Date:** 2026-09-02
**Repo:** `workshop-genkit-flutter-hybrid-ai` (DenisovAV)
**Status:** approved design, pending spec review → implementation plan

## Goal

Rewrite the workshop's **Step 4 "Hybrid Strategy"** to route on-device ↔ cloud
through the published **`genkit_hybrid` 0.2.0** package instead of the
hand-rolled `HybridAIService`, and add a new step that showcases all three
0.2.0 routing tools (`CapabilityStrategy`, `CostStrategy`, `cascadeModel`) as
**live app features**, including real image input.

## Why

`genkit_hybrid` 0.2.0 was built specifically to be the package this workshop
teaches. Today the workshop hand-rolls a `Stream<String>`-level router across
two separate `Genkit` instances; the package does the same job at the Genkit
`Model` level, in less code, and composes cleanly with RAG, streaming, and
multimodal. The migration turns "here's how you'd hand-roll it" into "here's the
package, and the hybrid is itself an ordinary `Model`".

## Global Constraints

- **Dependency floor — genkit 0.15.1 line** (all verified resolvable on pub.dev 2026-09-02):
  - `genkit: ^0.15.1` (was `^0.13.0`)
  - `genkit_google_genai: ^0.2.12` (was `^0.2.7`; 0.2.12 → genkit ^0.15.1)
  - `genkit_flutter_gemma: ^0.5.0` (was `^0.3.1`; 0.5.0 → genkit ^0.15.1, flutter_gemma ^1.5.9)
  - `flutter_gemma: ^1.7.0` (was `^0.15.1`; forced by genkit_flutter_gemma 0.5.0's `^1.5.9` floor)
  - **add** `genkit_hybrid: ^0.2.0`
  - **add** `image_picker: ^1.1.2` (verify latest at plan time)
- **No AI attribution** in commits / PR bodies (repo convention + maintainer rule); author `Sasha Denisov <denisov.shureg@gmail.com>`.
- **The codelab and the code must stay in lockstep** — every code change has a matching `codelab/index.md` change in the same work; the codelab IS the deliverable.
- **RAG (embeddings) keeps working** — it shares the single Genkit instance; do not regress Step 6.

## Non-goals (out of scope for this spec)

- **Workshop 1** (`workshop-flutter-gemma-hybrid-ai`, Firebase-AI) — analyzed separately (Firebase model-shutdown + App Check deadlines); NOT touched here.
- Restructuring RAG beyond re-wiring it to the new single-Genkit owner.
- Upgrading to genkit 0.16.0-rc (stay on the stable 0.15.1 line).
- Adding new cloud models or image generation.

## Architecture

### Before
Two `Genkit` instances: one in `CloudAIService` (googleAI plugin), one in
`LocalAIService` (flutter_gemma plugin, also shared with `RagService` for
embeddings). Each exposes `Stream<String> generateResponseStream(String)`.
`HybridAIService` routes between them at the string level with a manual
before-first-token fallback and an `AIStrategy {localFirst, cloudFirst,
localOnly, cloudOnly}` enum.

### After
**One `Genkit`** with both plugins, owned by a new thin `AiEngine`:

```dart
_ai = Genkit(plugins: [
  googleAI(apiKey: geminiApiKey),
  GenkitFlutterGemmaPlugin(
    models: [FlutterGemmaModelConfig(name: kLocalModel, modelType: ModelType.gemmaIt)],
    embedders: [FlutterGemmaEmbedderConfig(name: kEmbedder)],
  ),
]);
```

- Two base models: `local = flutterGemma.model(kLocalModel)`, `cloud = googleAI.gemini(kCloudModel)`.
- `AiEngine` owns init (`FlutterGemma.initialize` + `installModel` + `installEmbedder`), holds the `Genkit`, exposes `local`/`cloud`/the embedder, and **builds a genkit_hybrid `Model` for the selected policy** (see Strategies).
- `HybridAIService`, `CloudAIService`, `LocalAIService` are **deleted**; their responsibilities move into `AiEngine`. `AIService` abstract class is dropped (only one owner now).
- `RagService` is **unchanged internally**; it now receives the `AiEngine`'s Genkit + embedder name.
- `chat_screen` calls the composed hybrid model via `ai.generateStream(model: hybrid, messages: [...])`.

**Teaching narrative (new Step 4):** don't hand-roll a router — `genkit_hybrid`
gives a ready policy, and the hybrid is an ordinary `Model` composable with
everything else (RAG, streaming, multimodal).

## Strategies + UI

Replace the 4-way `SegmentedButton` with a **policy Dropdown** (5 modes; each
maps 1:1 to a single genkit_hybrid construct — the pedagogical point). Branch
keys use the package constants `kOnDevice` / `kCloud`.

| Policy | genkit_hybrid construct |
|---|---|
| **Cloud** | `hybridModel(branches, strategy: PreRoutingStrategy((_) => kCloud))` |
| **Local** | `PreRoutingStrategy((_) => kOnDevice)` |
| **Smart** | `WithFallback(FirstMatch([CapabilityStrategy(supports: {kOnDevice: {}, kCloud: {ModelCapability.vision}}), ConnectivityStrategy(isOnline:, online: kCloud, offline: kOnDevice)]), fallbackOrder: [kOnDevice])` |
| **Cascade** | `cascadeModel(branches, order: [kOnDevice, kCloud], accept: (r) => r.text.trim().length > 20)` |
| **Budget** | `hybridModel(branches, strategy: CostStrategy(budgetAvailable: () => _spent < _cap, premium: kCloud, cheap: kOnDevice))` — plus a small "budget spent / reset" control in the UI |

Notes:
- `Cloud`/`Local`/`Smart`/`Budget` are `hybridModel(...)` with different
  strategies. `Cascade` is a `cascadeModel(...)` — a different factory (not a
  strategy). `AiEngine` returns the right `Model` for the selected policy.
- The `budgetAvailable` signal and the `isOnline` signal are app-supplied
  closures (a spend counter, a connectivity check).

## Multimodal (image input)

- Add an image attach button (`image_picker`) to the chat input; the picked
  image rides on the outgoing message.
- The generate path shifts from a `String` prompt to Genkit **messages with a
  `MediaPart`**: `ai.generateStream(model: hybrid, messages: [Message.user([TextPart(text: prompt), if (image != null) MediaPart(media: Media(url: dataUri))])])`.
  (Confirm the exact `Message`/`Part` constructors against genkit 0.15.1 at
  plan time.)
- With an image attached, **Smart** mode's `CapabilityStrategy` sees the
  `vision` requirement and routes to **cloud** (local Gemma-3-1B is text-only).
- **Decision (recorded):** in the policies whose selected route starts on the
  text-only local model — **Local** and **Cascade** — the app **blocks send with
  an inline hint** ("The on-device model can't see images — switch to Smart or
  Cloud"). This is the live lesson for `CapabilityStrategy`, cleaner than
  silently overriding the user's chosen policy. **Cloud** accepts the image
  directly; **Smart** routes it to cloud automatically via `CapabilityStrategy`.

## Codelab structure

- **Step 4 "Hybrid Strategy" — rewritten**: build the single `Genkit` + both
  plugins, get the two base models, compose with `hybridModel` /
  `hybridModelOnDeviceCloud`, map the Cloud/Local/Local-first/Cloud-first modes
  to genkit_hybrid strategies, and land the new narrative ("the hybrid is a
  Model"). The old "routing is pure Dart above Genkit" punchline is replaced.
- **New Step 4.5 "Smart routing & images"**: add `image_picker`, the multimodal
  message path, and the three 0.2.0 tools as the Smart / Cascade / Budget
  policies, with the capability-block lesson.
- Other steps (Firebase-free cloud Step 2, local Step 3, embeddings Step 5, RAG
  Step 6, polish Step 7) — **minimal edits only** where the service rename or the
  message-path change ripples in (e.g. `RagService` construction, the
  `generateResponseStream` → `ai.generateStream` call site).
- Renumbering: keep it minimal — insert "Step 4.5" rather than renumber 5-7, or
  renumber if cleaner (plan decides).

## Error handling

- `genkit_hybrid` owns fallback semantics: transient errors fall to the next
  branch, **before-first-token only** (a partially streamed response is not
  re-routed) — same guarantee the hand-rolled code gave, now from the package.
- Missing `GEMINI_API_KEY` / failed local install: `AiEngine` surfaces which
  branches are available (as today `chat_screen` tracks `_cloudReady`/
  `_localReady`); policies needing an unavailable branch are disabled in the
  Dropdown.
- Image attached in a text-only policy: send blocked with the inline hint (above).

## Testing

The workshop is an app; `genkit_hybrid` itself is already unit-tested.
- `flutter analyze` clean, `flutter build` (at least one target) green.
- A small **pure-Dart smoke test** of `AiEngine`'s policy→construct mapping:
  assert that for a given (policy, request) the composed model / strategy yields
  the expected branch order (e.g. Smart + image → `[kCloud]`; Smart + text
  offline → `[kOnDevice]`; Budget spent → `[kOnDevice]`). No real models.
- Manual runbook in the codelab: text in each mode; image → cloud in Smart;
  disable network → fallback; spend budget → switches to local; image in Local →
  blocked with hint.

## Open items to resolve at plan time (not design blockers)

1. Exact genkit 0.15.1 `Message` / `TextPart` / `MediaPart` / `Media`
   constructors for the multimodal call and how `genkit_flutter_gemma` forwards
   (or rejects) media to a text-only model.
2. `flutter_gemma ^1.7.0` install API parity — the workshop already uses the
   1.x fluent `installModel().fromNetwork().withProgress().install()` /
   `installEmbedder().modelFromNetwork().tokenizerFromNetwork().install()`;
   confirm no signature drift at 1.7.0.
3. `image_picker` latest version + web support (the workshop lists `web/`).
4. How `googleAI.gemini('gemini-2.5-flash')` handles a `MediaPart` (vision) —
   confirm the cloud model id used supports vision (2.5-flash does).
