# genkit_hybrid Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the workshop's hybrid on-device↔cloud routing to use the `genkit_hybrid` 0.2.0 package (routing at the Genkit `Model` level) and add all three 0.2.0 tools as live features, including image input.

**Architecture:** One `Genkit` with both plugins (googleAI + flutter_gemma), owned by a new thin `AiEngine` that resolves the two base Models from the registry and composes them via `hybridModel`/`cascadeModel` per a selected `PolicyMode`. The old `Cloud/Local/HybridAIService` are deleted; `RagService` keeps the shared Genkit for embeddings; `chat_screen` gains a policy Dropdown, image attach, and a multimodal message path.

**Tech Stack:** Flutter, genkit 0.15.1, genkit_hybrid 0.2.0, genkit_flutter_gemma 0.5.0, genkit_google_genai 0.2.12, flutter_gemma 1.7.0, image_picker 1.2.3.

**Spec:** `docs/superpowers/specs/2026-09-02-genkit-hybrid-migration-design.md`

## Global Constraints

- **Dependency floor (genkit 0.15.1 line, all verified resolvable on pub.dev 2026-09-02):** `genkit ^0.15.1`, `genkit_google_genai ^0.2.12`, `genkit_flutter_gemma ^0.5.0`, `flutter_gemma ^1.7.0`, `genkit_hybrid ^0.2.0`, `image_picker ^1.2.3`.
- **`environment` must bump** to `sdk: '>=3.12.0 <4.0.0'`, `flutter: '>=3.44.0'` (flutter_gemma 1.7.0 floor).
- **No AI attribution** in commits / PR bodies; author `Sasha Denisov <denisov.shureg@gmail.com>`.
- **Verified genkit 0.15.1 API facts (use verbatim):**
  - No `Message.user()`. Build `Message(role: Role.user, content: [Part...])`. `Role` is an extension type: `Role.user`.
  - `TextPart({required String text})`; `MediaPart({required Media media})`; `Media({String? contentType, required String url})`.
  - `ai.generateStream(model: ModelRef?, prompt: String?, messages: List<Message>?)`; streamed chunk exposes `.text`.
  - **`Model extends Action implements ModelRef`** → resolve a plugin model to a concrete `Model` via `(await ai.registry.lookupAction('model', ref.name)) as Model`, using `ref.name` from `flutterGemma.model(name)` / `googleAI.gemini(id)` (no hardcoded prefixes).
- **flutter_gemma 1.7.0 install API (verified):** `installModel(modelType:).fromNetwork(url, token:).withProgress((int p){}).install()` — **`withProgress` callback is `int` 0-100, not double**; `installEmbedder().modelFromNetwork(url, token:).tokenizerFromNetwork(url, token:).install()`.
- **genkit_flutter_gemma media caveat:** its converter **silently drops** a `MediaPart` whose `Media.contentType` is null or does not start with `image`/`audio`. Always set `Media(contentType: mime, url: dataUri)`. A media part in a `system` message throws — only put media in `Role.user` messages.
- **genkit_hybrid strategy signatures (verified):** `PreRoutingStrategy(String Function(RoutingContext) select)` (return `''` = abstain); `FirstMatch(List<RoutingStrategy> children)`; `WithFallback(RoutingStrategy inner, {required List<String> fallbackOrder})`; `ConnectivityStrategy({required bool Function() isOnline, required String online, required String offline})`; `CapabilityStrategy({required Map<String, Set<ModelCapability>> supports, List<String>? order})`; `CostStrategy({required bool Function() budgetAvailable, required String premium, required String cheap})`; `hybridModel({required Map<String, Model> branches, required RoutingStrategy strategy, String name})`; `cascadeModel({required Map<String, Model> branches, required List<String> order, required FutureOr<bool> Function(ModelResponse) accept, String name})`. Branch keys: `kOnDevice`, `kCloud`.

### Plan-time correction to the spec's "Smart" composition

The spec proposed `Smart = WithFallback(FirstMatch([CapabilityStrategy, ConnectivityStrategy]), fallbackOrder:[kOnDevice])`. **This does not work:** `CapabilityStrategy` returns *all capable branches* (non-empty) for a text request (both branches satisfy the empty requirement), so `FirstMatch` stops at it and never consults `ConnectivityStrategy`. Corrected, package-only composition that preserves the intent (image→cloud; text→cloud-first with offline handled by fallback):

```dart
Smart = WithFallback(
  CapabilityStrategy(supports: {kCloud: {ModelCapability.vision}, kOnDevice: <ModelCapability>{}}),
  fallbackOrder: [kOnDevice],
);
```
`kCloud` is listed first in `supports` so a text request routes cloud-first with the on-device model as the transient-failure fallback (an offline cloud call throws → `hybridModel` falls to `kOnDevice` automatically — no explicit connectivity signal needed). An image request → capability yields `[kCloud]` only → `[kCloud, kOnDevice]` after the fallback tail. `ConnectivityStrategy` stays documented in the codelab strategy table but is not wired into Smart. (Package-feedback note, non-blocking: a future `CapabilityStrategy` "abstain when no capability is required" option would let the original FirstMatch composition work.)

## File Structure

- **Modify** `pubspec.yaml` — deps + environment.
- **Create** `lib/services/ai_engine.dart` — `PolicyMode` enum + `AiEngine` (single Genkit, both plugins, model resolution, policy→strategy/model builders, budget/capability helpers).
- **Delete** `lib/services/ai_service.dart`, `cloud_ai_service.dart`, `local_ai_service.dart`, `hybrid_ai_service.dart`.
- **Modify** `lib/models/message_model.dart` — optional image fields.
- **Modify** `lib/screens/chat_screen.dart` — `AiEngine` wiring, policy Dropdown, image attach, multimodal message path, capability-block.
- **Modify** `lib/services/rag_service.dart` — none internally; only its construction in `chat_screen` changes (now from `AiEngine`).
- **Create** `test/ai_engine_policy_test.dart` — pure-Dart routing smoke test.
- **Modify** `codelab/index.md` — Step 4 rewrite + new Step 4.5 + ripple edits.

---

## Task 1: Dependencies & environment

**Files:** Modify `pubspec.yaml`.

- [ ] **Step 1: Bump `environment` and dependencies.** Replace the `environment` block and the `dependencies` (keep `flutter`, `cupertino_icons`):

```yaml
environment:
  sdk: '>=3.12.0 <4.0.0'
  flutter: '>=3.44.0'

dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.8

  # Cloud AI via Gemini API (Genkit)
  genkit: ^0.15.1
  genkit_google_genai: ^0.2.12

  # On-device AI via genkit_flutter_gemma
  genkit_flutter_gemma: ^0.5.0
  flutter_gemma: ^1.7.0

  # Hybrid on-device ↔ cloud routing
  genkit_hybrid: ^0.2.0

  # Image input (multimodal)
  image_picker: ^1.2.3
```

- [ ] **Step 2: Resolve and analyze.**

Run: `cd /Users/sashadenisov/Work/workshop-genkit-flutter-hybrid-ai && flutter pub get`
Expected: resolves with no version conflict.
Run: `flutter analyze`
Expected: 0 errors. (The old services still compile against genkit 0.15.1 / flutter_gemma 1.7.0 — their `ai.generateStream(model:, prompt:)`, `chunk.text`, and the install chain are unchanged. If a real drift appears, fix it minimally in place — those files are deleted in Task 2 anyway.)

- [ ] **Step 3: Commit.**

```bash
git add pubspec.yaml pubspec.lock
git commit --author="Sasha Denisov <denisov.shureg@gmail.com>" -m "chore: bump to genkit 0.15.1 line + add genkit_hybrid, image_picker"
```

---

## Task 2: AiEngine + policy routing + smoke test

**Files:**
- Create: `lib/services/ai_engine.dart`
- Create: `test/ai_engine_policy_test.dart`
- Delete: `lib/services/ai_service.dart`, `lib/services/cloud_ai_service.dart`, `lib/services/local_ai_service.dart`, `lib/services/hybrid_ai_service.dart`

**Interfaces:**
- Produces: `enum PolicyMode { cloud, local, smart, cascade, budget }`; `class AiEngine` with `Future<void> initialize({void Function(int)? onProgress})`, `Genkit get ai`, `String get embedderName`, `bool cloudReady`, `bool localReady`, `int cloudCallsSpent`, `int budgetCap`, `bool get budgetAvailable`, `Model modelFor(PolicyMode)`, `RoutingStrategy strategyFor(PolicyMode)`, `bool requiresTextOnly(PolicyMode)`, `Future<void> dispose()`; consts `kLocalModel`, `kCloudModel`, `kEmbedder`.

- [ ] **Step 1: Write the failing routing test.** Create `test/ai_engine_policy_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_hybrid/genkit_hybrid.dart';
import 'package:workshop_genkit_flutter_hybrid_ai/services/ai_engine.dart';

RoutingContext _ctx({bool withImage = false}) => RoutingContext(
      request: ModelRequest(messages: [
        Message(role: Role.user, content: [
          TextPart(text: 'hi'),
          if (withImage)
            MediaPart(
              media: Media(contentType: 'image/png', url: 'data:image/png;base64,AA'),
            ),
        ]),
      ]),
      branchKeys: const {kOnDevice, kCloud},
      isStreaming: false,
    );

void main() {
  test('cloud mode always routes to cloud', () {
    expect(AiEngine().strategyFor(PolicyMode.cloud).route(_ctx()), [kCloud]);
  });

  test('local mode always routes to on-device', () {
    expect(AiEngine().strategyFor(PolicyMode.local).route(_ctx()), [kOnDevice]);
  });

  test('smart mode: text is cloud-first with on-device fallback', () {
    expect(AiEngine().strategyFor(PolicyMode.smart).route(_ctx()),
        [kCloud, kOnDevice]);
  });

  test('smart mode: an image is routed to cloud only (capability), plus fallback', () {
    // The inner CapabilityStrategy excludes the text-only on-device model when
    // an image is present; WithFallback then appends it as the safety tail.
    final route = AiEngine().strategyFor(PolicyMode.smart).route(_ctx(withImage: true));
    expect(route.first, kCloud);
  });

  test('budget mode: premium while budget holds, cheap once spent', () {
    final e = AiEngine()
      ..budgetCap = 2
      ..cloudCallsSpent = 0;
    expect(e.strategyFor(PolicyMode.budget).route(_ctx()), [kCloud, kOnDevice]);
    e.cloudCallsSpent = 2;
    expect(e.strategyFor(PolicyMode.budget).route(_ctx()), [kOnDevice]);
  });

  test('requiresTextOnly is true only for local and cascade', () {
    final e = AiEngine();
    expect(e.requiresTextOnly(PolicyMode.local), isTrue);
    expect(e.requiresTextOnly(PolicyMode.cascade), isTrue);
    expect(e.requiresTextOnly(PolicyMode.smart), isFalse);
    expect(e.requiresTextOnly(PolicyMode.cloud), isFalse);
  });
}
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `flutter test test/ai_engine_policy_test.dart`
Expected: FAIL to compile — `ai_engine.dart` does not exist yet.

- [ ] **Step 3: Create `lib/services/ai_engine.dart`.**

```dart
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';
import 'package:genkit_hybrid/genkit_hybrid.dart';

const _modelUrl =
    'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task';
const _embeddingModelUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite';
const _tokenizerUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model';

// Pass at build time: --dart-define=HF_TOKEN=hf_xxx --dart-define=GEMINI_API_KEY=AIza...
const _hfToken = String.fromEnvironment('HF_TOKEN');
const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');

const kLocalModel = 'gemma-3-1b-it';
const kCloudModel = 'gemini-2.5-flash';
const kEmbedder = 'embedding-gemma-300m';

/// The five routing policies the chat exposes. Each maps to one genkit_hybrid
/// construct (see [modelFor] / [strategyFor]).
enum PolicyMode { cloud, local, smart, cascade, budget }

/// Owns a single Genkit instance with both plugins (cloud + on-device),
/// resolves the two base models, and composes them via genkit_hybrid per the
/// selected [PolicyMode]. Replaces the old Cloud/Local/HybridAIService trio.
class AiEngine {
  Genkit? _ai;
  Model? _local;
  Model? _cloud;

  bool cloudReady = false;
  bool localReady = false;

  // CostStrategy demo signal: the app counts cloud calls against a small cap.
  int cloudCallsSpent = 0;
  int budgetCap = 3;
  bool get budgetAvailable => cloudCallsSpent < budgetCap;

  Genkit get ai {
    final ai = _ai;
    if (ai == null) throw StateError('AiEngine not initialized');
    return ai;
  }

  String get embedderName => kEmbedder;

  Future<void> initialize({void Function(int progress)? onProgress}) async {
    final plugins = <GenkitPlugin>[];

    if (_geminiApiKey.isNotEmpty) {
      plugins.add(googleAI(apiKey: _geminiApiKey));
      cloudReady = true;
    }

    await FlutterGemma.initialize();
    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromNetwork(_modelUrl, token: _hfToken.isEmpty ? null : _hfToken)
        .withProgress((p) => onProgress?.call(p)) // p is int 0..100
        .install();
    await FlutterGemma.installEmbedder()
        .modelFromNetwork(_embeddingModelUrl,
            token: _hfToken.isEmpty ? null : _hfToken)
        .tokenizerFromNetwork(_tokenizerUrl,
            token: _hfToken.isEmpty ? null : _hfToken)
        .install();
    plugins.add(GenkitFlutterGemmaPlugin(
      models: [
        FlutterGemmaModelConfig(name: kLocalModel, modelType: ModelType.gemmaIt),
      ],
      embedders: [FlutterGemmaEmbedderConfig(name: kEmbedder)],
    ));
    localReady = true;

    _ai = Genkit(plugins: plugins);
    _local = await _resolve(flutterGemma.model(kLocalModel));
    if (cloudReady) _cloud = await _resolve(googleAI.gemini(kCloudModel));
  }

  // A plugin model is registered by name; genkit's `Model` is an `Action`, so
  // look the concrete model up from the registry and cast.
  Future<Model> _resolve(ModelRef ref) async {
    final action = await ai.registry.lookupAction('model', ref.name);
    if (action == null) {
      throw StateError('model "${ref.name}" is not registered');
    }
    return action as Model;
  }

  Map<String, Model> get _branches => {
        if (_local != null) kOnDevice: _local!,
        if (_cloud != null) kCloud: _cloud!,
      };

  /// The composable Genkit `Model` for [mode]. Cascade is a `cascadeModel`;
  /// every other mode is `hybridModel(strategy: strategyFor(mode))`.
  Model modelFor(PolicyMode mode) {
    if (mode == PolicyMode.cascade) {
      return cascadeModel(
        branches: _branches,
        order: const [kOnDevice, kCloud],
        accept: (r) => r.text.trim().length > 20,
        name: 'cascade',
      );
    }
    return hybridModel(
      branches: _branches,
      strategy: strategyFor(mode),
      name: mode.name,
    );
  }

  /// Pure policy → RoutingStrategy mapping (no models needed), so the routing
  /// decisions are unit-testable. [PolicyMode.cascade] has no strategy — it is
  /// a Model, built in [modelFor].
  RoutingStrategy strategyFor(PolicyMode mode) {
    switch (mode) {
      case PolicyMode.cloud:
        return PreRoutingStrategy((_) => kCloud);
      case PolicyMode.local:
        return PreRoutingStrategy((_) => kOnDevice);
      case PolicyMode.smart:
        // Image → cloud (only it declares vision). Text → cloud-first (kCloud
        // listed first), on-device as the transient-failure fallback (an
        // offline cloud call throws → hybridModel falls to on-device).
        return WithFallback(
          CapabilityStrategy(supports: {
            kCloud: {ModelCapability.vision},
            kOnDevice: <ModelCapability>{},
          }),
          fallbackOrder: const [kOnDevice],
        );
      case PolicyMode.budget:
        return CostStrategy(
          budgetAvailable: () => budgetAvailable,
          premium: kCloud,
          cheap: kOnDevice,
        );
      case PolicyMode.cascade:
        throw ArgumentError('cascade has no RoutingStrategy; use modelFor');
    }
  }

  /// True when [mode]'s primary route starts on the text-only on-device model,
  /// so an attached image cannot be handled (used to block send with a hint).
  bool requiresTextOnly(PolicyMode mode) =>
      mode == PolicyMode.local || mode == PolicyMode.cascade;

  Future<void> dispose() async {
    _ai = null;
    _local = null;
    _cloud = null;
  }
}
```

- [ ] **Step 4: Delete the old services.**

```bash
git rm lib/services/ai_service.dart lib/services/cloud_ai_service.dart \
       lib/services/local_ai_service.dart lib/services/hybrid_ai_service.dart
```
(They are replaced by `AiEngine`. `chat_screen.dart` still imports them — that break is fixed in Task 3; `flutter analyze` will error on `chat_screen` until then, which is expected between tasks.)

- [ ] **Step 5: Run the routing test.**

Run: `flutter test test/ai_engine_policy_test.dart`
Expected: PASS (6 tests). This exercises `strategyFor` + `requiresTextOnly` with no real models. If `lookupAction`'s actionType string is not `'model'` it does not affect this test (the test never calls `initialize`); it is verified live in Task 3 / the final runbook.

- [ ] **Step 6: Commit.**

```bash
git add lib/services/ai_engine.dart test/ai_engine_policy_test.dart
git commit --author="Sasha Denisov <denisov.shureg@gmail.com>" -m "feat: AiEngine — genkit_hybrid routing over one Genkit + both plugins"
```

---

## Task 3: message_model + chat_screen wiring (text path)

**Files:**
- Modify: `lib/models/message_model.dart`
- Modify: `lib/screens/chat_screen.dart`

**Interfaces:**
- Consumes: `AiEngine`, `PolicyMode`, `kOnDevice`/`kCloud` (Task 2); `RagService` (unchanged).
- Produces: a working text chat driven by `AiEngine.modelFor(policy)` via `ai.generateStream(model:, messages:)`.

- [ ] **Step 1: Add optional image to `ChatMessage`.** Replace `lib/models/message_model.dart`:

```dart
import 'dart:typed_data';

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final Uint8List? imageBytes; // set on a user message that attached an image

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
```

- [ ] **Step 2: Rewrite `chat_screen.dart` state to use `AiEngine`.** Replace the imports + field block + `initState`/`_initServices` region. Key changes (full replacements):

Imports (drop the deleted services):
```dart
import 'package:flutter/material.dart';
import '../models/message_model.dart';
import '../services/ai_engine.dart';
import '../services/rag_service.dart';
import '../widgets/message_bubble.dart';
```

Fields — replace the three service fields + `_strategy` with:
```dart
  late final AiEngine _engine;
  RagService? _ragService;

  PolicyMode _policy = PolicyMode.cloud;
```
(Keep `_messages`, `_isGenerating`, `_isInitializing`, `_downloadProgress`, `_statusMessage`, `_cloudReady`, `_localReady`, `_ragReady`, `_ragEnabled`, `_lastRagSources`, the UI-throttle fields, `_controller`, `_scrollController`.)

`initState` + `_initServices`:
```dart
  @override
  void initState() {
    super.initState();
    _engine = AiEngine();
    _initServices();
  }

  Future<void> _initServices() async {
    try {
      if (mounted) setState(() => _statusMessage = 'Downloading local model...');
      await _engine.initialize(
        onProgress: (p) {
          if (mounted) setState(() => _downloadProgress = p / 100);
        },
      );
    } catch (e) {
      debugPrint('AiEngine init failed: $e');
    }

    _cloudReady = _engine.cloudReady;
    _localReady = _engine.localReady;

    if (_localReady) {
      try {
        if (mounted) setState(() => _statusMessage = 'Setting up RAG...');
        final rag = RagService(ai: _engine.ai, embedderName: _engine.embedderName);
        await rag.initialize(
          onStatus: (s) { if (mounted) setState(() => _statusMessage = s); },
        );
        _ragService = rag;
        _ragReady = true;
      } catch (e) {
        debugPrint('RAG init failed: $e');
      }
    }

    if (!mounted) return;
    final defaultPolicy = switch ((_cloudReady, _localReady)) {
      (true, _) => PolicyMode.cloud,
      (false, true) => PolicyMode.local,
      _ => PolicyMode.cloud,
    };
    final parts = [
      if (_cloudReady) 'cloud',
      if (_localReady) 'local',
      if (_ragReady) 'RAG',
    ];
    setState(() {
      _isInitializing = false;
      _policy = defaultPolicy;
      _statusMessage = parts.isEmpty ? 'No services available' : '${parts.join(', ')} ready';
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    _engine.dispose();
    _ragService?.dispose();
    super.dispose();
  }
```

- [ ] **Step 3: Route generation through the engine.** Replace the body of `_sendMessage`'s generation section — build a text-only user message and stream from the composed model. Replace the `final buffer = StringBuffer();` … `await for (final chunk in _hybridService.generateResponseStream(prompt))` block with:

```dart
      final userMessage = Message(
        role: Role.user,
        content: [TextPart(text: prompt)],
      );

      final buffer = StringBuffer();
      _lastUiUpdate = DateTime.now();

      final stream = _engine.ai.generateStream(
        model: _engine.modelFor(_policy),
        messages: [userMessage],
      );
      await for (final chunk in stream) {
        buffer.write(chunk.text);
        final now = DateTime.now();
        if (now.difference(_lastUiUpdate) >= _uiUpdateInterval) {
          _lastUiUpdate = now;
          if (!mounted) return;
          setState(() {
            _messages.last = ChatMessage(text: buffer.toString(), isUser: false);
          });
          _scrollToBottom();
        }
      }
      if (_policy == PolicyMode.budget || _policy == PolicyMode.cloud) {
        _engine.cloudCallsSpent++; // demo accounting for CostStrategy
      }
```
Add the genkit import at the top of the file: `import 'package:genkit/genkit.dart';`. Keep the RAG-augmentation block above it (it produces `prompt`), the trailing final `setState`, and the `finally` placeholder-cleanup unchanged.

- [ ] **Step 4: Replace the strategy `SegmentedButton` with a policy `Dropdown`.** Replace the `SegmentedButton<AIStrategy>(...)` widget with:

```dart
            DropdownButton<PolicyMode>(
              value: _policy,
              isExpanded: true,
              items: [
                DropdownMenuItem(value: PolicyMode.cloud, enabled: _cloudReady, child: const Text('Cloud')),
                DropdownMenuItem(value: PolicyMode.local, enabled: _localReady, child: const Text('Local')),
                DropdownMenuItem(value: PolicyMode.smart, enabled: _cloudReady && _localReady, child: const Text('Smart (image-aware)')),
                DropdownMenuItem(value: PolicyMode.cascade, enabled: _cloudReady && _localReady, child: const Text('Cascade (escalate on quality)')),
                DropdownMenuItem(value: PolicyMode.budget, enabled: _cloudReady && _localReady, child: const Text('Budget (cost-gated)')),
              ],
              onChanged: (m) { if (m != null) setState(() => _policy = m); },
            ),
```
Delete `_onStrategyChanged`. (Image attach + capability-block come in Task 4.)

- [ ] **Step 5: Analyze + build.**

Run: `flutter analyze`
Expected: 0 errors (all references to the deleted services are gone).
Run: `flutter build web --no-tree-shake-icons` (or another available target)
Expected: builds.

- [ ] **Step 6: Commit.**

```bash
git add lib/models/message_model.dart lib/screens/chat_screen.dart
git commit --author="Sasha Denisov <denisov.shureg@gmail.com>" -m "feat: chat drives AiEngine.modelFor(policy) via genkit messages; policy dropdown"
```

---

## Task 4: Image input + multimodal + capability-block

**Files:** Modify `lib/screens/chat_screen.dart`.

**Interfaces:**
- Consumes: `AiEngine.requiresTextOnly` (Task 2); `image_picker` (Task 1).
- Produces: an attach-image button; a multimodal user message; a send-block with hint for text-only policies.

- [ ] **Step 1: Add image state + picker.** Add imports and fields:
```dart
import 'dart:typed_data';
import 'package:image_picker/image_picker.dart';
```
Fields:
```dart
  final _picker = ImagePicker();
  Uint8List? _attachedImage;
  String? _attachedMime;
```
Method:
```dart
  Future<void> _attachImage() async {
    final XFile? picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;
    final bytes = await picked.readAsBytes();
    if (!mounted) return;
    setState(() {
      _attachedImage = bytes;
      _attachedMime = picked.mimeType ?? 'image/jpeg';
    });
  }
```

- [ ] **Step 2: Block send in text-only policies when an image is attached.** At the top of `_sendMessage`, after computing `text`:
```dart
    if (_attachedImage != null && _engine.requiresTextOnly(_policy)) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text("The on-device model can't see images — switch to Smart or Cloud."),
      ));
      return;
    }
```

- [ ] **Step 3: Build a multimodal message.** In `_sendMessage`, replace the `userMessage` construction (Task 3, Step 3) with:
```dart
      final content = <Part>[TextPart(text: prompt)];
      if (_attachedImage != null) {
        final mime = _attachedMime ?? 'image/jpeg';
        final dataUri = 'data:$mime;base64,${base64Encode(_attachedImage!)}';
        // contentType MUST be set: the on-device plugin drops media without an
        // image/* contentType, and CapabilityStrategy reads it to detect vision.
        content.add(MediaPart(media: Media(contentType: mime, url: dataUri)));
      }
      final userMessage = Message(role: Role.user, content: content);
```
Add `import 'dart:convert';` for `base64Encode`. Also stamp the attached image onto the user `ChatMessage` (in the earlier `setState` that adds the user bubble): `ChatMessage(text: text, isUser: true, imageBytes: _attachedImage)`, and clear it after send: in the `finally`, `_attachedImage = null; _attachedMime = null;`.

- [ ] **Step 4: Add the attach button + a thumbnail chip to the input row.** Before the `TextField`'s `IconButton.filled` send button, add an attach button:
```dart
                  IconButton(
                    onPressed: (_isGenerating || _isInitializing) ? null : _attachImage,
                    icon: const Icon(Icons.image_outlined),
                    tooltip: 'Attach image',
                  ),
```
And above the input `Container`, when `_attachedImage != null`, show a small preview with a remove button:
```dart
          if (_attachedImage != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Row(children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.memory(_attachedImage!, width: 40, height: 40, fit: BoxFit.cover),
                ),
                const SizedBox(width: 8),
                const Text('Image attached', style: TextStyle(fontSize: 12)),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.close, size: 18),
                  onPressed: () => setState(() { _attachedImage = null; _attachedMime = null; }),
                ),
              ]),
            ),
```
(Optionally render `message.imageBytes` in `MessageBubble` — nice-to-have; not required for the lesson.)

- [ ] **Step 5: Analyze + build + manual check.**

Run: `flutter analyze` → 0 errors. `flutter build web --no-tree-shake-icons` → builds.
**Runtime verify (needs GEMINI_API_KEY + HF_TOKEN + a device/emulator):** in Smart mode, attach an image, send → the cloud model describes the image (confirms `googleAI.gemini('gemini-2.5-flash')` accepts a `MediaPart` — the one runtime-verified integration point). In Local/Cascade, attaching + send shows the block snackbar.

- [ ] **Step 6: Commit.**

```bash
git add lib/screens/chat_screen.dart
git commit --author="Sasha Denisov <denisov.shureg@gmail.com>" -m "feat: image input + multimodal message; capability-block for text-only policies"
```

---

## Task 5: Codelab rewrite

**Files:** Modify `codelab/index.md`.

**Interfaces:** Consumes the final code from Tasks 1-4; the codelab code blocks must match the shipped code.

- [ ] **Step 1: Rewrite Step 4 "Hybrid Strategy".** Replace the current Step 4 body (the hand-rolled `HybridAIService` + "routing is pure Dart" narrative) with: (a) add the deps (`genkit_hybrid`, and note the 0.15.1 bump); (b) build the single `Genkit` with both plugins inside `AiEngine`; (c) resolve the two base models via `registry.lookupAction('model', ref.name) as Model`; (d) map Cloud/Local to `PreRoutingStrategy`, and introduce `hybridModel(branches:, strategy:)`; (e) the new punchline: **the hybrid is itself an ordinary `Model`** — you `ai.generateStream(model: engine.modelFor(policy), messages: ...)`, and it composes with RAG/streaming/images. Paste the real `AiEngine` (Task 2) and the `_sendMessage` streaming call (Task 3) as the code blocks. **Use only ` ```dart ` or plain fences.**

- [ ] **Step 2: Add new Step 4.5 "Smart routing & images".** Cover: the multimodal message path (Task 4), `image_picker`, and the three 0.2.0 tools as the Smart / Cascade / Budget policies — with the `strategyFor` code, the CapabilityStrategy note (image→cloud; the text-only on-device model is excluded), the cascade `accept` predicate, the CostStrategy budget signal, and the capability-block UX. Include the manual runbook (text in each mode; image→cloud in Smart; disable network → Smart falls back to local; spend the budget → Budget switches to local; image in Local → blocked).

- [ ] **Step 3: Ripple edits.** Update Step 3 (local inference) if it referenced `LocalAIService` — it now lives inside `AiEngine`. Update Step 6 (RAG) where it constructed `RagService(ai: _localService.ai, ...)` → `RagService(ai: _engine.ai, embedderName: _engine.embedderName)`. Fix any Step numbering that references "Step 5"→ the embeddings step if a 4.5 insertion shifts it; prefer inserting "Step 4.5" without renumbering 5-7.

- [ ] **Step 4: Verify the codelab is buildable prose.** Grep for non-Dart fences:
Run: `grep -rhoE '```[a-z]+' codelab/index.md | sort | uniq -c`
Expected: only ` ```dart ` (plus plain fences). No `yaml`/`bash`/`xml` fences (keep pubspec snippets in plain fences).

- [ ] **Step 5: Commit.**

```bash
git add codelab/index.md
git commit --author="Sasha Denisov <denisov.shureg@gmail.com>" -m "docs(codelab): Step 4 rewrite on genkit_hybrid + new Step 4.5 smart routing & images"
```

---

## Self-Review notes (author)

- **Spec coverage:** Architecture (Task 2), 5 modes (Task 2 `strategyFor`/`modelFor`), multimodal + capability-block (Task 4), deps (Task 1), codelab (Task 5), RAG preserved (Task 3). Testing = the pure-Dart routing smoke test (Task 2) + build/analyze per task + the manual runbook (Task 5).
- **The one runtime-verified point:** `gemini-2.5-flash` accepting a `MediaPart` via `genkit_google_genai` (Task 4, Step 5) — the package is not installed locally so this is confirmed on device, not from source. Text generation via `googleAI.gemini(...)` is already proven by the pre-migration app.
- **`lookupAction('model', …)` actionType** `'model'` mirrors the verified `lookupAction('embedder', …)`; confirmed live the first time `initialize()` runs (Task 3 build/run). If it differs, it is a one-string fix in `AiEngine._resolve`.
- **Smart composition** corrected from the spec (see Global Constraints) — package-only, image→cloud, offline via fallback.
