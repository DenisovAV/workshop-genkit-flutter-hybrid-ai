import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';
import 'ai_service.dart';

const String _modelUrl =
    'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task';
const String _embeddingModelUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite';
const String _tokenizerUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model';

// Pass at build time: flutter run --dart-define=HF_TOKEN=hf_xxx
const String _hfToken = String.fromEnvironment('HF_TOKEN');

const String _modelName = 'gemma-3-1b-it';
const String _embedderName = 'embedding-gemma-300m';

class LocalAIService implements AIService {
  Genkit? _ai;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  // Shared Genkit instance exposed for RagService to use for embeddings.
  Genkit get ai {
    final ai = _ai;
    if (ai == null) throw StateError('LocalAIService not initialized');
    return ai;
  }

  String get embedderName => _embedderName;

  @override
  Future<void> initialize({void Function(double)? onProgress}) async {
    if (_isInitialized) return;

    await FlutterGemma.initialize();

    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromNetwork(_modelUrl, token: _hfToken.isNotEmpty ? _hfToken : null)
        .withProgress((p) => onProgress?.call(p / 100))
        .install();

    await FlutterGemma.installEmbedder()
        .modelFromNetwork(
          _embeddingModelUrl,
          token: _hfToken.isNotEmpty ? _hfToken : null,
        )
        .tokenizerFromNetwork(
          _tokenizerUrl,
          token: _hfToken.isNotEmpty ? _hfToken : null,
        )
        .install();

    // One Genkit instance for both inference and embeddings.
    _ai = Genkit(plugins: [
      GenkitFlutterGemmaPlugin(
        models: [
          FlutterGemmaModelConfig(
            name: _modelName,
            modelType: ModelType.gemmaIt,
          ),
        ],
        embedders: [FlutterGemmaEmbedderConfig(name: _embedderName)],
      ),
    ]);

    _isInitialized = true;
  }

  @override
  Stream<String> generateResponseStream(String prompt) async* {
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
