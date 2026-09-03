import 'package:flutter/foundation.dart' show debugPrint, visibleForTesting;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit/plugin.dart' show GenkitPlugin;
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';
import 'package:genkit_hybrid/genkit_hybrid.dart';

// Prod installs the on-device LLM straight from Hugging Face by repo + file
// (the plugin applies the configured token to gated huggingface.co URLs).
const _hfRepo = 'litert-community/Gemma3-1B-IT';
const _hfModelFile = 'Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm';
const _embeddingModelUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite';
const _tokenizerUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model';

// Pass at build time: --dart-define=HF_TOKEN=hf_xxx --dart-define=GEMINI_API_KEY=AIza...
const _hfToken = String.fromEnvironment('HF_TOKEN');
const _geminiApiKey = String.fromEnvironment('GEMINI_API_KEY');

const kLocalModel = 'gemma-3-1b-it';
const kCloudModel = 'gemini-3.7-flash';
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

  /// One composite [Model] per policy, built AND registered once by
  /// [_registerPolicyModels] — genkit reduces `generate(model: ...)` to the
  /// model's name and looks it up in the registry, so a freshly built,
  /// never-registered composite would fail every call with NOT_FOUND.
  final Map<PolicyMode, Model> _models = {};

  bool cloudReady = false;
  bool localReady = false;

  // CostStrategy demo signal: the app counts cloud calls against a small cap.
  int cloudCallsSpent = 0;
  int budgetCap = 3;
  bool get budgetAvailable => cloudCallsSpent < budgetCap;

  AiEngine();

  /// Test seam: skips [FlutterGemma.initialize]/`installModel` (real I/O that
  /// can't run in a unit test) and takes already-resolved branch models
  /// directly, then runs the same build+register path [initialize] uses — so
  /// a test driving [modelFor] through `ai.generate` here exercises the real
  /// registration wiring, not just [strategyFor]. [local] is nullable —
  /// mirrors [cloud] — so a cloud-only engine (the symmetric mirror of the
  /// cloud-absent case) is constructible too.
  @visibleForTesting
  AiEngine.forTest({required Genkit ai, Model? local, Model? cloud}) {
    _ai = ai;
    _local = local;
    _cloud = cloud;
    localReady = local != null;
    cloudReady = cloud != null;
    _registerPolicyModels();
  }

  Genkit get ai {
    final ai = _ai;
    if (ai == null) throw StateError('AiEngine not initialized');
    return ai;
  }

  String get embedderName => kEmbedder;

  Future<void> initialize({
    void Function(int progress)? onProgress,
    // Test seam: install the LLM from a pre-staged local file instead of
    // downloading it — avoids a flaky ~500MB on-device download on CI / FTL.
    String? localModelPath,
    // Test seam: skip the embedder download when RAG isn't exercised.
    bool downloadEmbedder = true,
  }) async {
    // Declarative plugin config — always includes the on-device plugin (its
    // models/embedders are looked up by name later, independent of whether
    // the install below actually succeeds).
    final plugins = <GenkitPlugin>[
      if (_geminiApiKey.isNotEmpty) googleAI(apiKey: _geminiApiKey),
      GenkitFlutterGemmaPlugin(
        models: [
          FlutterGemmaModelConfig(
            name: kLocalModel,
            modelType: ModelType.gemmaIt,
            fileType: ModelFileType.litertlm,
          ),
        ],
        embedders: downloadEmbedder
            ? [FlutterGemmaEmbedderConfig(name: kEmbedder)]
            : const [],
      ),
    ];

    // Build Genkit BEFORE any on-device engine registration/install so `_ai`
    // (and `_resolve`, and the `ai` getter) are always available afterward —
    // the plugin list above is purely declarative (no I/O, no dependency on
    // FlutterGemma.initialize() having run), so cloud resolution below needs
    // no on-device engine and must not be taken down by a failure
    // registering/installing it.
    _ai = Genkit(plugins: plugins);

    // CLOUD: needs no install, so its readiness never depends on the local
    // LLM or the (optional) embedder below.
    if (_geminiApiKey.isNotEmpty) {
      try {
        _cloud = await _resolve(googleAI.gemini(kCloudModel));
        cloudReady = true;
      } catch (e) {
        debugPrint('⚠️ AiEngine: CLOUD backend unavailable — $e');
        cloudReady = false;
      }
    }

    // LOCAL: register the on-device engine, then install + resolve the LLM.
    // flutter_gemma 1.x registers no engines by default; that registration
    // now lives inside this try/catch (not before Genkit is built) so an
    // engine-init failure only suppresses localReady, never cloud.
    try {
      // Opt into LiteRT-LM (.litertlm inference) + its LiteRT embedding
      // backend.
      await FlutterGemma.initialize(
        inferenceEngines: [LiteRtLmEngine()],
        embeddingBackends: [LiteRtEmbeddingBackend()],
      );

      // fileType MUST be litertlm — installModel defaults to task (MediaPipe),
      // which no registered engine would handle here.
      final llm = FlutterGemma.installModel(
        modelType: ModelType.gemmaIt,
        fileType: ModelFileType.litertlm,
      );
      if (localModelPath != null) {
        await llm.fromFile(localModelPath).install();
      } else {
        await llm
            .fromHuggingFace(
              _hfRepo,
              file: _hfModelFile,
              token: _hfToken.isEmpty ? null : _hfToken,
            )
            .withProgress((p) => onProgress?.call(p)) // p is int 0..100
            .install();
      }
      _local = await _resolve(flutterGemma.model(kLocalModel));
      localReady = true;
    } catch (e) {
      debugPrint('⚠️ AiEngine: on-device backend unavailable — $e');
      localReady = false;
    }

    // EMBEDDER (OPTIONAL): RAG-only, never blocks chat — a failure here must
    // not flip localReady or rethrow.
    if (downloadEmbedder && localReady) {
      try {
        await FlutterGemma.installEmbedder()
            .modelFromNetwork(
              _embeddingModelUrl,
              token: _hfToken.isEmpty ? null : _hfToken,
            )
            .tokenizerFromNetwork(
              _tokenizerUrl,
              token: _hfToken.isEmpty ? null : _hfToken,
            )
            .install();
      } catch (e) {
        debugPrint('⚠️ AiEngine: embedder install failed, RAG disabled — $e');
      }
    }

    _registerPolicyModels();
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

  /// Builds AND registers one composite [Model] per [PolicyMode] whose
  /// required branches are available, populating [_models]. A mode that needs
  /// `kCloud` (every mode but `local`) is skipped when there's no API key, so
  /// a missing cloud branch degrades to a clear [modelFor] error instead of
  /// crashing here on a half-built `cascadeModel` (its `order` validates
  /// eagerly against `branches`, unlike `hybridModel`).
  void _registerPolicyModels() {
    final branches = _branches;
    for (final mode in PolicyMode.values) {
      if (!_hasRequiredBranches(mode, branches)) continue;
      final model = _buildModel(mode);
      ai.registry.register(model);
      _models[mode] = model;
    }
  }

  bool _hasRequiredBranches(PolicyMode mode, Map<String, Model> branches) {
    switch (mode) {
      case PolicyMode.cloud:
        return branches.containsKey(kCloud);
      case PolicyMode.local:
        return branches.containsKey(kOnDevice);
      case PolicyMode.smart:
      case PolicyMode.cascade:
      case PolicyMode.budget:
        return branches.containsKey(kOnDevice) && branches.containsKey(kCloud);
    }
  }

  /// The composable Genkit `Model` for [mode]. Cascade is a `cascadeModel`;
  /// every other mode is `hybridModel(strategy: strategyFor(mode))`. Called
  /// once per mode by [_registerPolicyModels] — not a per-request factory.
  Model _buildModel(PolicyMode mode) {
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

  /// The registered, resolvable `Model` for [mode] — built once by
  /// [_registerPolicyModels] during [initialize] (or [AiEngine.forTest]).
  Model modelFor(PolicyMode mode) {
    final model = _models[mode];
    if (model == null) {
      throw StateError(
        'No model registered for $mode — call initialize() first (or, if '
        'this mode needs the cloud branch, make sure a cloud API key is set).',
      );
    }
    return model;
  }

  /// Pure policy → RoutingStrategy mapping (no models needed), so the routing
  /// decisions are unit-testable. [PolicyMode.cascade] has no strategy — it is
  /// a Model, built in [_buildModel].
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
          CapabilityStrategy(
            supports: {
              kCloud: {ModelCapability.vision},
              kOnDevice: <ModelCapability>{},
            },
          ),
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
    _models.clear();
  }
}
