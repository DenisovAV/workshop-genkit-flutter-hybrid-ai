author: Sasha Denisov
summary: Hybrid AI in Flutter with Genkit Dart — From Cloud to On-Device
id: hybrid-ai-flutter-genkit
categories: flutter, ai, gemma, genkit
environments: web, android, ios
status: Published

# Hybrid AI in Flutter: From Cloud to On-Device with Genkit Dart

## Overview
Duration: 5

### What you'll build

A Flutter chat application that progressively integrates AI capabilities using Genkit Dart:

1. **Cloud Chat** — Streaming responses from Gemini via `genkit_google_genai`
2. **Local Inference** — On-device AI with Gemma 3 1B via `genkit_flutter_gemma`
3. **Hybrid Strategy** — Automatic fallback between cloud and local
4. **Embeddings** — Semantic vector representations with EmbeddingGemma via Genkit
5. **RAG** — Context-augmented generation using a local tourist guide

### What you'll learn

- How to use the Genkit Dart framework for AI inference in Flutter
- How to route between cloud and on-device models using a single `Genkit` instance
- How to run AI models locally on device with `genkit_flutter_gemma`
- How text embeddings work and how to build a RAG pipeline with Genkit

### What you'll need

- Flutter 3.27+ installed
- A GEMINI_API_KEY from [aistudio.google.com](https://aistudio.google.com)
- A HuggingFace account (for model downloads)
- Android device/emulator, iOS simulator, or macOS
- ~1 GB free disk space (for the AI model)

### Architecture

```
┌──────────────────────────────────────────┐
│              Flutter App                 │
├──────────────────────────────────────────┤
│           HybridAIService                │
├──────────────────┬───────────────────────┤
│  CloudAIService  │    LocalAIService     │
│  Genkit instance │    Genkit instance    │
│  googleAI plugin │ GenkitFlutterGemma    │
│                  │       plugin          │
├──────────────────┼───────────────────────┤
│  gemini-2.5-     │  Gemma 3 1B           │
│  flash (cloud)   │  + EmbeddingGemma     │
│                  │  (on-device)          │
└──────────────────┴───────────────────────┘
```

The key insight: both cloud and on-device inference use the same Genkit API —
`ai.generateStream(model: ..., prompt: ...)`. The only thing that changes is
the `model:` parameter.

## Step 1: Starter Project
Duration: 5

### Clone the repository

```bash
git clone https://github.com/DenisovAV/workshop-genkit-flutter-hybrid-ai.git
cd workshop-genkit-flutter-hybrid-ai
git checkout step-00-starter
flutter pub get
```

### Explore the project

Open the project in your IDE. The starter includes:

- **`lib/main.dart`** — Simple app entry point, no async setup needed
- **`lib/screens/chat_screen.dart`** — Chat UI with TextField, ListView, send button
- **`lib/widgets/message_bubble.dart`** — Styled message bubbles (user right, AI left)
- **`lib/models/message_model.dart`** — Simple `ChatMessage` data class
- **`lib/services/ai_service.dart`** — Abstract `AIService` interface
- **`assets/tourist_data/`** — 10 JSON files with city descriptions (Paris, Tokyo, New York…)

### The AIService interface

All our AI services implement this contract:

```dart
abstract class AIService {
  Future<void> initialize();
  Stream<String> generateResponseStream(String prompt);
  Future<void> dispose();
}
```

`generateResponseStream` returns a `Stream<String>` — responses stream
token-by-token for a real-time chat feel.

### Run the starter

```bash
flutter run
```

You'll see the chat UI. Messages echo back with a placeholder. Let's replace
that with real AI next.

## Step 2: Cloud Chat with Genkit
Duration: 15

### Get your API key

Go to [aistudio.google.com](https://aistudio.google.com), sign in with your
Google account, and click **Get API key**. Copy the key — you'll use it with
`--dart-define`.

> No Firebase project, no CLI setup — just an API key.

### Update dependencies

In `pubspec.yaml`, uncomment the Genkit dependencies:

```yaml
dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.8

  # Step 2: Cloud AI
  genkit: ^0.13.0
  genkit_google_genai: ^0.2.7
```

Run `flutter pub get`.

### Create CloudAIService

Create `lib/services/cloud_ai_service.dart`:

```dart
import 'package:genkit/genkit.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';
import 'ai_service.dart';

// Pass at build time: flutter run --dart-define=GEMINI_API_KEY=AIza...
const String _apiKey = String.fromEnvironment('GEMINI_API_KEY');

class CloudAIService implements AIService {
  Genkit? _ai;

  @override
  Future<void> initialize() async {
    if (_apiKey.isEmpty) {
      throw StateError('GEMINI_API_KEY is not set. '
          'Run with --dart-define=GEMINI_API_KEY=your_key');
    }
    _ai = Genkit(plugins: [googleAI(apiKey: _apiKey)]);
  }

  @override
  Stream<String> generateResponseStream(String prompt) async* {
    final ai = _ai;
    if (ai == null) throw StateError('CloudAIService not initialized');

    final stream = ai.generateStream(
      model: googleAI.gemini('gemini-2.5-flash'),
      prompt: prompt,
    );

    await for (final chunk in stream) {
      if (chunk.text.isNotEmpty) yield chunk.text;
    }
  }

  @override
  Future<void> dispose() async => _ai = null;
}
```

### Wire it up in chat_screen.dart

Replace the echo stub:

```dart
import '../services/cloud_ai_service.dart';

// in _ChatScreenState:
late final CloudAIService _cloudService;

// in initState:
_cloudService = CloudAIService();
_initServices();

// in _initServices:
await _cloudService.initialize();

// in _sendMessage:
await for (final chunk in _cloudService.generateResponseStream(prompt)) {
  buffer.write(chunk);
  // ... update UI
}
```

### Run with your API key

```bash
flutter run --dart-define=GEMINI_API_KEY=your_key_here
```

Type "Tell me about Paris" — Gemini streams a response token by token.

> **What happened?** `Genkit(plugins: [googleAI(...)])` registered Gemini as a
> model provider. `ai.generateStream(model: googleAI.gemini('gemini-2.5-flash'), ...)`
> streams the response. The `Genkit` instance is the single point of contact
> for all AI operations.

## Step 3: Local Inference with genkit_flutter_gemma
Duration: 20

### Platform setup

**iOS** (`ios/Runner/Info.plist`):
```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Required for local model download</string>
```
Minimum deployment target: iOS 16+.

**Android** (`android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

### Update dependencies

Add `genkit_flutter_gemma` and `flutter_gemma`:

```yaml
  # Step 3: On-device AI
  genkit_flutter_gemma: ^0.3.1
  flutter_gemma: ^0.15.1
```

Run `flutter pub get`.

### Get a HuggingFace token

Go to [huggingface.co](https://huggingface.co), sign in, and create a
read-access token at **Settings → Access Tokens**.

### Create LocalAIService

Create `lib/services/local_ai_service.dart`:

```dart
import 'package:flutter/material.dart' show WidgetsFlutterBinding;
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:genkit/genkit.dart';
import 'package:genkit_flutter_gemma/genkit_flutter_gemma.dart';
import 'ai_service.dart';

const String _modelUrl =
    'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task';
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

    // Download the model (skipped if already installed)
    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromNetwork(_modelUrl, token: _hfToken.isNotEmpty ? _hfToken : null)
        .withProgress((p) => onProgress?.call(p / 100))
        .install();

    // Register the model with Genkit
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
```

### Run with HuggingFace token

```bash
flutter run --dart-define=GEMINI_API_KEY=your_key --dart-define=HF_TOKEN=hf_xxx
```

The first run downloads ~600 MB. Subsequent runs use the cached model.

> **Key insight**: Notice that `generateResponseStream` looks identical to
> `CloudAIService` — only the `model:` parameter changes. Genkit decouples
> _what model to use_ from _how to call it_.

```dart
// Cloud:
ai.generateStream(model: googleAI.gemini('gemini-2.5-flash'), prompt: prompt)

// Local:
ai.generateStream(model: flutterGemma.model('gemma-3-1b-it'), prompt: prompt)
```

Same API. Different backends.

## Step 4: Hybrid Strategy
Duration: 15

### Create HybridAIService

Create `lib/services/hybrid_ai_service.dart`:

```dart
import 'ai_service.dart';
import 'cloud_ai_service.dart';
import 'local_ai_service.dart';

enum AIStrategy { localFirst, cloudFirst, localOnly, cloudOnly }

class HybridAIService implements AIService {
  final LocalAIService local;
  final CloudAIService cloud;

  AIStrategy strategy = AIStrategy.localFirst;

  HybridAIService({required this.local, required this.cloud});

  @override
  Future<void> initialize() async {
    await cloud.initialize();
    await local.initialize();
  }

  @override
  Stream<String> generateResponseStream(String prompt) async* {
    switch (strategy) {
      case AIStrategy.localFirst:
        var yielded = false;
        try {
          await for (final chunk in local.generateResponseStream(prompt)) {
            yielded = true;
            yield chunk;
          }
        } catch (e) {
          if (yielded) rethrow; // don't silently switch mid-response
          yield* cloud.generateResponseStream(prompt);
        }
      case AIStrategy.cloudFirst:
        var yielded = false;
        try {
          await for (final chunk in cloud.generateResponseStream(prompt)) {
            yielded = true;
            yield chunk;
          }
        } catch (e) {
          if (yielded) rethrow;
          yield* local.generateResponseStream(prompt);
        }
      case AIStrategy.localOnly:
        yield* local.generateResponseStream(prompt);
      case AIStrategy.cloudOnly:
        yield* cloud.generateResponseStream(prompt);
    }
  }

  @override
  Future<void> dispose() async {
    await local.dispose();
    await cloud.dispose();
  }
}
```

### Add strategy picker to the UI

In `chat_screen.dart`, add a `SegmentedButton<AIStrategy>` with four options:
Cloud Only, Local Only, Local→Cloud, Cloud→Local.

### Test the fallback

1. Switch to **Local→Cloud**
2. Disable WiFi
3. Send a message — the local model responds
4. Re-enable WiFi — cloud responds again

> **Notice**: `HybridAIService` doesn't know anything about Genkit. It routes
> between two `AIService` implementations. Genkit lives _inside_ each service.
> The routing logic — `switch (strategy)` with `yield*` — is pure Dart.

## Step 5: Embeddings with Genkit
Duration: 20

### How embeddings work

An embedding turns text into a vector of numbers that captures semantic meaning.
Similar texts have similar vectors. EmbeddingGemma 300M runs entirely on-device.

### Install the embedding model

```dart
await FlutterGemma.installEmbedder()
    .modelFromNetwork(embeddingModelUrl, token: token)
    .tokenizerFromNetwork(tokenizerUrl, token: token)
    .install();
```

### Register the embedder with Genkit

```dart
_ai = Genkit(plugins: [
  GenkitFlutterGemmaPlugin(
    models: const [],
    embedders: [FlutterGemmaEmbedderConfig(name: 'embedding-gemma-300m')],
  ),
]);
```

### Generate embeddings

```dart
final embeddings = await ai.embed(
  embedder: flutterGemma.embedder('embedding-gemma-300m'),
  document: DocumentData(content: [TextPart(text: content)]),
);
final vector = embeddings.first.embedding; // List<double>
```

### Index the tourist data

Create `lib/services/rag_service.dart`. In `initialize()`, loop over 10 city
JSON files and embed each one. Store the `List<double>` vectors in memory.

```dart
for (final city in _cityFiles) {
  final content = _buildContent(data); // description + attractions + cuisine...
  final embeddings = await ai.embed(
    embedder: flutterGemma.embedder(_embedderName),
    document: DocumentData(content: [TextPart(text: content)]),
  );
  _store.add(_VectorDocument(id: city, content: content,
      city: name, embedding: embeddings.first.embedding));
}
```

> **Key insight**: `ai.embed(embedder: ...)` is model-agnostic. Replace
> `flutterGemma.embedder(...)` with any other registered embedder — the rest
> of the code stays the same.

## Step 6: RAG — Retrieval-Augmented Generation
Duration: 20

### Semantic search

When the user sends a query, embed it and find the closest city documents
using cosine similarity:

```dart
Future<RagResult> searchAndBuildContext(String query) async {
  // 1. Embed the query
  final queryEmbeddings = await _ai!.embed(
    embedder: flutterGemma.embedder(_embedderName),
    document: DocumentData(content: [TextPart(text: query)]),
  );
  final queryVector = queryEmbeddings.first.embedding;

  // 2. Score all documents
  final scored = _store
      .map((doc) => (doc: doc, score: _cosine(queryVector, doc.embedding)))
      .where((r) => r.score >= 0.5)
      .toList()
    ..sort((a, b) => b.score.compareTo(a.score));

  // 3. Build augmented prompt with top 3 results
  final topK = scored.take(3);
  final context = topK.map((r) => r.doc.content).join('\n\n');
  final augmentedPrompt =
      'Based on the following travel information:\n\n$context\n\n'
      'Answer the question: $query';

  return RagResult(augmentedPrompt: augmentedPrompt, ...);
}
```

### Add RAG toggle to the UI

In `chat_screen.dart`:
1. Add a `Switch` in the `AppBar` to toggle RAG
2. In `_sendMessage()`, if RAG is enabled call `searchAndBuildContext(text)` before generating
3. Display `ragResult.sources` in a banner below the AppBar

### Test it

Try these queries:
- "What should I eat in Tokyo?" → sources: Tokyo (92%)
- "Best European city for history?" → sources: Prague (78%), Istanbul (71%)
- "Tell me about the Eiffel Tower" → sources: Paris (95%)

## Step 7: Polish and Conclusion
Duration: 10

### Error handling and loading states

- Show download progress during model installation
- Disable strategy buttons when a service fails to initialize
- Show "Generating..." indicator during streaming
- Graceful error messages in the chat

### What we built

| Capability | Technology |
|------------|-----------|
| Cloud inference | `genkit_google_genai` → Gemini 2.5 Flash |
| On-device inference | `genkit_flutter_gemma` → Gemma 3 1B |
| Hybrid routing | `HybridAIService` + 4 strategies |
| On-device embeddings | `genkit_flutter_gemma` → EmbeddingGemma 300M |
| RAG pipeline | Genkit `embed()` + in-memory cosine search |

### The Genkit advantage

The old approach needed two completely different APIs — Firebase AI Logic for
cloud and raw flutter_gemma calls for local. With Genkit:

```dart
// Both use the same API — only model: changes
ai.generateStream(model: googleAI.gemini('gemini-2.5-flash'), prompt: prompt)
ai.generateStream(model: flutterGemma.model('gemma-3-1b-it'), prompt: prompt)
ai.embed(embedder: flutterGemma.embedder('embedding-gemma-300m'), document: ...)
```

Routing, fallback, observability, and tool calling all work the same way
regardless of which model backend you use.

### What's next

- **Production embeddings**: Persist the vector store with SQLite + `drift` so
  you don't re-embed on every cold start
- **More models**: Swap `gemini-2.5-flash` for `gemini-2.5-pro` for complex
  queries, or add a second on-device model for specialized tasks
- **Genkit flows**: Wrap the hybrid routing in a `defineFlow` to add
  observability, retries, and structured output
- **On-device RAG with Qdrant**: Replace the in-memory store with
  [Qdrant](https://qdrant.tech) for persistent, scalable vector search

### Resources

- [genkit_flutter_gemma on pub.dev](https://pub.dev/packages/genkit_flutter_gemma)
- [genkit on pub.dev](https://pub.dev/packages/genkit)
- [genkit_google_genai on pub.dev](https://pub.dev/packages/genkit_google_genai)
- [flutter_gemma on pub.dev](https://pub.dev/packages/flutter_gemma)
- [Genkit Dart documentation](https://genkit.dev)
