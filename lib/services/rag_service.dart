import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';

const String _embeddingModelUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite';
const String _tokenizerUrl =
    'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/sentencepiece.model';

// Pass at build time: flutter run --dart-define=HF_TOKEN=hf_xxx
const String _hfToken = String.fromEnvironment('HF_TOKEN');

const String _embedderName = 'embedding-gemma-300m';

const List<String> _cityFiles = [
  'paris',
  'tokyo',
  'new_york',
  'barcelona',
  'istanbul',
  'sydney',
  'rio',
  'marrakech',
  'prague',
  'singapore',
];

class _VectorDocument {
  final String id;
  final String content;
  final String city;
  final List<double> embedding;

  const _VectorDocument({
    required this.id,
    required this.content,
    required this.city,
    required this.embedding,
  });
}

class RagService {
  Genkit? _ai;
  final List<_VectorDocument> _store = [];
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;
  int get documentCount => _store.length;

  Future<void> initialize({void Function(String status)? onStatus}) async {
    onStatus?.call('Installing embedding model...');
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

    _ai = Genkit(plugins: [
      GenkitFlutterGemmaPlugin(
        models: const [],
        embedders: [FlutterGemmaEmbedderConfig(name: _embedderName)],
      ),
    ]);

    onStatus?.call('Loading tourist data...');
    await _loadTouristData(onStatus);

    _isInitialized = true;
    onStatus?.call('RAG ready (${_store.length} documents)');
  }

  Future<void> _loadTouristData(void Function(String)? onStatus) async {
    final ai = _ai!;
    for (final city in _cityFiles) {
      onStatus?.call('Embedding $city...');
      final jsonString =
          await rootBundle.loadString('assets/tourist_data/$city.json');
      final data = jsonDecode(jsonString) as Map<String, dynamic>;

      final name = data['name'] as String? ?? city;
      final content = _buildContent(data);

      final embeddings = await ai.embed(
        embedder: flutterGemma.embedder(_embedderName),
        document: DocumentData(content: [TextPart(text: content)]),
      );

      _store.add(_VectorDocument(
        id: city,
        content: content,
        city: name,
        embedding: embeddings.first.embedding,
      ));
    }
  }

  String _buildContent(Map<String, dynamic> data) {
    final description = data['description'] as String? ?? '';
    final attractions = ((data['attractions'] as List?) ?? []).join(', ');
    final cuisine = ((data['cuisine'] as List?) ?? []).join(', ');
    final bestTime = data['best_time'] as String? ?? '';
    final funFact = data['fun_fact'] as String? ?? '';
    return '$description\n\n'
        'Top attractions: $attractions.\n'
        'Local cuisine: $cuisine.\n'
        'Best time to visit: $bestTime.\n'
        'Fun fact: $funFact';
  }

  Future<RagResult> searchAndBuildContext(String query) async {
    if (!_isInitialized) throw StateError('RagService not initialized');

    final queryEmbeddings = await _ai!.embed(
      embedder: flutterGemma.embedder(_embedderName),
      document: DocumentData(content: [TextPart(text: query)]),
    );
    final queryVector = queryEmbeddings.first.embedding;

    final scored = _store
        .map((doc) => (doc: doc, score: _cosine(queryVector, doc.embedding)))
        .where((r) => r.score >= 0.5)
        .toList()
      ..sort((a, b) => b.score.compareTo(a.score));

    final topK = scored.take(3).toList();

    if (topK.isEmpty) {
      return RagResult(augmentedPrompt: query, retrievedContext: '', sources: []);
    }

    final context = topK.map((r) => r.doc.content).join('\n\n');
    final sources = topK
        .map((r) =>
            '${r.doc.city} (${(r.score * 100).toStringAsFixed(0)}%)')
        .toList();

    final augmentedPrompt =
        'Based on the following travel information:\n\n$context\n\n'
        'Answer the question: $query';

    return RagResult(
      augmentedPrompt: augmentedPrompt,
      retrievedContext: context,
      sources: sources,
    );
  }

  double _cosine(List<double> a, List<double> b) {
    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    final denom = math.sqrt(normA) * math.sqrt(normB);
    return denom == 0 ? 0.0 : dot / denom;
  }

  Future<void> dispose() async {
    _store.clear();
    _ai = null;
    _isInitialized = false;
  }
}

class RagResult {
  final String augmentedPrompt;
  final String retrievedContext;
  final List<String> sources;

  const RagResult({
    required this.augmentedPrompt,
    required this.retrievedContext,
    required this.sources,
  });

  bool get hasContext => retrievedContext.isNotEmpty;
}
