import 'package:flutter/material.dart' show WidgetsFlutterBinding;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';
import 'ai_service.dart';

const String _modelUrl =
    'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task';

// Pass at build time: flutter run --dart-define=HF_TOKEN=hf_xxx
const String _hfToken = String.fromEnvironment('HF_TOKEN');

const String _modelName = 'gemma-3-1b-it';

class LocalAIService implements AIService {
  Genkit? _ai;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  @override
  Future<void> initialize({void Function(double)? onProgress}) async {
    WidgetsFlutterBinding.ensureInitialized();
    await FlutterGemma.initialize();

    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromNetwork(_modelUrl, token: _hfToken.isNotEmpty ? _hfToken : null)
        .withProgress((p) => onProgress?.call(p / 100))
        .install();

    _ai = Genkit(plugins: [
      GenkitFlutterGemmaPlugin(
        models: [
          FlutterGemmaModelConfig(
            name: _modelName,
            modelType: ModelType.gemmaIt,
          ),
        ],
      ),
    ]);

    _isInitialized = true;
  }

  @override
  Stream<String> generateResponseStream(String prompt) async* {
    final ai = _ai;
    if (ai == null) throw StateError('LocalAIService not initialized');

    final stream = ai.generateStream(
      model: flutterGemma.model(_modelName),
      prompt: prompt,
    );

    await for (final chunk in stream) {
      if (chunk.text.isNotEmpty) yield chunk.text;
    }
  }

  @override
  Future<void> dispose() async {
    _ai = null;
    _isInitialized = false;
  }
}
