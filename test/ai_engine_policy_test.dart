import 'package:flutter_test/flutter_test.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_hybrid/genkit_hybrid.dart';
import 'package:workshop_genkit_flutter_hybrid_ai/services/ai_engine.dart';

RoutingContext _ctx({bool withImage = false}) => RoutingContext(
  request: ModelRequest(
    messages: [
      Message(
        role: Role.user,
        content: [
          TextPart(text: 'hi'),
          if (withImage)
            MediaPart(
              media: Media(
                contentType: 'image/png',
                url: 'data:image/png;base64,AA',
              ),
            ),
        ],
      ),
    ],
  ),
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
    expect(AiEngine().strategyFor(PolicyMode.smart).route(_ctx()), [
      kCloud,
      kOnDevice,
    ]);
  });

  test(
    'smart mode: an image is routed to cloud only (capability), plus fallback',
    () {
      // The inner CapabilityStrategy excludes the text-only on-device model when
      // an image is present; WithFallback then appends it as the safety tail.
      final route = AiEngine()
          .strategyFor(PolicyMode.smart)
          .route(_ctx(withImage: true));
      expect(route.first, kCloud);
    },
  );

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

  // --------------------------------------------------------------------
  // End-to-end: modelFor -> ai.generate. This is the wiring `strategyFor`
  // alone never exercises — `strategyFor(mode).route(ctx)` only returns a
  // branch key, it never touches the registry. Before the fix, `modelFor`
  // built a fresh, never-registered `hybridModel`/`cascadeModel` on every
  // call; `ai.generate` reduces `model:` to `model.name` and looks it up via
  // `registry.lookupAction('model', name)`, so every real send threw
  // `GenkitException('Model <name> not found', NOT_FOUND)`. These tests fail
  // against that code and pass once `modelFor` returns a model that was
  // registered by `AiEngine`'s build+register path (see `AiEngine.forTest`).
  group('modelFor -> ai.generate (registration wiring)', () {
    Model fakeBranch(String name, String text) => Model(
      name: name,
      fn: (request, context) async => ModelResponse(
        finishReason: FinishReason.stop,
        message: Message(
          role: Role.model,
          content: [TextPart(text: text)],
        ),
      ),
    );

    test('cloud policy: modelFor is registered and routes to cloud', () async {
      final ai = Genkit(isDevEnv: false);
      final engine = AiEngine.forTest(
        ai: ai,
        local: fakeBranch('flutter-gemma/local', 'LOCAL'),
        cloud: fakeBranch('googleai/cloud', 'CLOUD'),
      );

      final resp = await ai.generate(
        model: engine.modelFor(PolicyMode.cloud),
        prompt: 'hi',
      );

      expect(resp.text, 'CLOUD');
    });

    test(
      'local policy: modelFor is registered and routes to on-device',
      () async {
        final ai = Genkit(isDevEnv: false);
        final engine = AiEngine.forTest(
          ai: ai,
          local: fakeBranch('flutter-gemma/local', 'LOCAL'),
          cloud: fakeBranch('googleai/cloud', 'CLOUD'),
        );

        final resp = await ai.generate(
          model: engine.modelFor(PolicyMode.local),
          prompt: 'hi',
        );

        expect(resp.text, 'LOCAL');
      },
    );
  });
}
